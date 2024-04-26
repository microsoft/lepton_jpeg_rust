/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use anyhow::{Context, Result};

use std::cmp;
use std::io::Write;

use crate::consts::UNZIGZAG_49;
use crate::enabled_features::EnabledFeatures;
use crate::helpers::*;
use crate::lepton_error::ExitCode;

use crate::metrics::Metrics;
use crate::structs::{
    block_based_image::AlignedBlock, block_based_image::BlockBasedImage,
    block_context::BlockContext, model::Model, model::ModelPerColor,
    neighbor_summary::NeighborSummary, probability_tables::ProbabilityTables,
    probability_tables_set::ProbabilityTablesSet, quantization_tables::QuantizationTables,
    row_spec::RowSpec, truncate_components::*, vpx_bool_writer::VPXBoolWriter,
};

use default_boxed::DefaultBoxed;

use super::block_context::NeighborData;

#[inline(never)] // don't inline so that the profiler can get proper data
pub fn lepton_encode_row_range<W: Write>(
    pts: &ProbabilityTablesSet,
    quantization_tables: &[QuantizationTables],
    image_data: &[BlockBasedImage],
    writer: &mut W,
    _thread_id: i32,
    colldata: &TruncateComponents,
    min_y: i32,
    max_y: i32,
    is_last_thread: bool,
    full_file_compression: bool,
    features: &EnabledFeatures,
) -> Result<Metrics> {
    let mut model = Model::default_boxed();
    let mut bool_writer = VPXBoolWriter::new(writer)?;

    let mut is_top_row = Vec::new();
    let mut neighbor_summary_cache = Vec::new();

    // Init helper structures
    for i in 0..image_data.len() {
        is_top_row.push(true);

        let num_non_zeros_length = (image_data[i].get_block_width() << 1) as usize;

        let mut neighbor_summary_component = Vec::new();
        neighbor_summary_component.resize(num_non_zeros_length, NeighborSummary::new());

        neighbor_summary_cache.push(neighbor_summary_component);
    }

    let component_size_in_blocks = colldata.get_component_sizes_in_blocks();
    let max_coded_heights = colldata.get_max_coded_heights();

    let mut encode_index = 0;
    loop {
        let cur_row = RowSpec::get_row_spec_from_index(
            encode_index,
            image_data,
            colldata.mcu_count_vertical,
            &max_coded_heights,
        );
        encode_index += 1;

        if cur_row.done {
            break;
        }

        if cur_row.luma_y >= max_y && !(is_last_thread && full_file_compression) {
            break;
        }

        if cur_row.skip {
            continue;
        }

        if cur_row.luma_y < min_y {
            continue;
        }

        // Advance to next row to cache expended block data for current row. Should be called before getting block context.
        let bt = cur_row.component;

        let mut block_context = image_data[bt].off_y(cur_row.curr_y);

        let block_width = image_data[bt].get_block_width();

        if is_top_row[bt] {
            is_top_row[bt] = false;
            process_row(
                &mut model,
                &mut bool_writer,
                &image_data[bt],
                &quantization_tables[bt],
                &pts.corner[bt],
                &pts.top[bt],
                &pts.top[bt],
                colldata,
                &mut block_context,
                &mut neighbor_summary_cache[bt][..],
                block_width,
                component_size_in_blocks[bt],
                features,
            )
            .context(here!())?;
        } else if block_width > 1 {
            process_row(
                &mut model,
                &mut bool_writer,
                &image_data[bt],
                &quantization_tables[bt],
                &pts.mid_left[bt],
                &pts.middle[bt],
                &pts.mid_right[bt],
                colldata,
                &mut block_context,
                &mut neighbor_summary_cache[bt][..],
                block_width,
                component_size_in_blocks[bt],
                features,
            )
            .context(here!())?;
        } else {
            assert!(block_width == 1, "block_width == 1");
            process_row(
                &mut model,
                &mut bool_writer,
                &image_data[bt],
                &quantization_tables[bt],
                &pts.width_one[bt],
                &pts.width_one[bt],
                &pts.width_one[bt],
                colldata,
                &mut block_context,
                &mut neighbor_summary_cache[bt][..],
                block_width,
                component_size_in_blocks[bt],
                features,
            )
            .context(here!())?;
        }
    }

    if is_last_thread && full_file_compression {
        let test = RowSpec::get_row_spec_from_index(
            encode_index,
            image_data,
            colldata.mcu_count_vertical,
            &max_coded_heights,
        );

        assert!(
            test.skip && test.done,
            "Row spec test: cmp {0} luma {1} item {2} skip {3} done {4}",
            test.component,
            test.luma_y,
            test.curr_y,
            test.skip,
            test.done
        );
    }

    bool_writer.finish().context(here!())?;

    Ok(bool_writer.drain_stats())
}

#[inline(never)] // don't inline so that the profiler can get proper data
fn process_row<W: Write>(
    model: &mut Model,
    bool_writer: &mut VPXBoolWriter<W>,
    image_data: &BlockBasedImage,
    qt: &QuantizationTables,
    left_model: &ProbabilityTables,
    middle_model: &ProbabilityTables,
    right_model: &ProbabilityTables,
    _colldata: &TruncateComponents,
    state: &mut BlockContext,
    neighbor_summary_cache: &mut [NeighborSummary],
    block_width: i32,
    component_size_in_block: i32,
    features: &EnabledFeatures,
) -> Result<()> {
    if block_width > 0 {
        serialize_tokens::<W, false>(
            state,
            qt,
            left_model,
            model,
            image_data,
            neighbor_summary_cache,
            bool_writer,
            features,
        )
        .context(here!())?;
        let offset = state.next(true);

        if offset >= component_size_in_block {
            return Ok(());
        }
    }

    for _jpeg_x in 1..block_width - 1 {
        // shortcut all the checks for the presence of left/right components by passing a constant generic parameter
        if middle_model.is_all_present() {
            serialize_tokens::<W, true>(
                state,
                qt,
                middle_model,
                model,
                image_data,
                neighbor_summary_cache,
                bool_writer,
                features,
            )
            .context(here!())?;
        } else {
            serialize_tokens::<W, false>(
                state,
                qt,
                middle_model,
                model,
                image_data,
                neighbor_summary_cache,
                bool_writer,
                features,
            )
            .context(here!())?;
        }

        let offset = state.next(true);

        if offset >= component_size_in_block {
            return Ok(());
        }
    }

    if block_width > 1 {
        if right_model.is_all_present() {
            serialize_tokens::<W, true>(
                state,
                qt,
                right_model,
                model,
                image_data,
                neighbor_summary_cache,
                bool_writer,
                features,
            )
            .context(here!())?;
        } else {
            serialize_tokens::<W, false>(
                state,
                qt,
                right_model,
                model,
                image_data,
                neighbor_summary_cache,
                bool_writer,
                features,
            )
            .context(here!())?;
        }

        state.next(false);
    }
    Ok(())
}

#[inline(never)] // don't inline so that the profiler can get proper data
fn serialize_tokens<W: Write, const ALL_PRESENT: bool>(
    context: &mut BlockContext,
    qt: &QuantizationTables,
    pt: &ProbabilityTables,
    model: &mut Model,
    image_data: &BlockBasedImage,
    neighbor_summary_cache: &mut [NeighborSummary],
    bool_writer: &mut VPXBoolWriter<W>,
    features: &EnabledFeatures,
) -> Result<()> {
    debug_assert!(ALL_PRESENT == pt.is_all_present());

    let block = context.here(image_data);

    let neighbors =
        context.get_neighbor_data::<ALL_PRESENT>(image_data, neighbor_summary_cache, pt);

    #[cfg(feature = "detailed_tracing")]
    trace!(
        "block {0}:{1:x}",
        context.get_here_index(),
        block.get_hash()
    );

    let ns = write_coefficient_block::<ALL_PRESENT, W>(
        pt,
        &neighbors,
        block,
        model,
        bool_writer,
        qt,
        features,
    )?;

    context.set_neighbor_summary_here(neighbor_summary_cache, ns);

    Ok(())
}

/// Writes the 8x8 coefficient block to the bit writer, taking into account the neighboring
/// blocks, probability tables and model.
///
/// This function is designed to be independently callable without needing to know the context,
/// image data, etc so it can be extensively unit tested.
pub fn write_coefficient_block<const ALL_PRESENT: bool, W: Write>(
    pt: &ProbabilityTables,
    neighbors_data: &NeighborData,
    here: &AlignedBlock,
    model: &mut Model,
    bool_writer: &mut VPXBoolWriter<W>,
    qt: &QuantizationTables,
    features: &EnabledFeatures,
) -> Result<NeighborSummary> {
    let model_per_color = model.get_per_color(pt);

    // first we encode the 49 inner coefficients
    let num_non_zeros_7x7 = here.get_count_of_non_zeros_7x7();

    let predicted_num_non_zeros_7x7 =
        pt.calc_non_zero_counts_context_7x7::<ALL_PRESENT>(neighbors_data);

    model_per_color
        .write_non_zero_7x7_count(bool_writer, predicted_num_non_zeros_7x7, num_non_zeros_7x7)
        .context(here!())?;

    // these are used as predictors for the number of non-zero edge coefficients
    let mut eob_x = 0;
    let mut eob_y = 0;

    let mut num_non_zeros_left_7x7 = num_non_zeros_7x7;

    let best_priors = pt.calc_coefficient_context_7x7_aavg_block::<ALL_PRESENT>(
        &neighbors_data.left,
        &neighbors_data.above,
        &neighbors_data.above_left,
    );

    for zig49 in 0..49 {
        if num_non_zeros_left_7x7 == 0 {
            break;
        }

        let coord = UNZIGZAG_49[zig49];

        let best_prior_bit_length = u16_bit_length(best_priors[coord as usize] as u16);

        // this should work in all cases but doesn't utilize that the zig49 is related
        let coef = here.get_coefficient(coord as usize);

        model_per_color
            .write_coef(
                bool_writer,
                coef,
                zig49,
                ProbabilityTables::num_non_zeros_to_bin_7x7(num_non_zeros_left_7x7) as usize,
                best_prior_bit_length as usize,
            )
            .context(here!())?;

        if coef != 0 {
            num_non_zeros_left_7x7 -= 1;

            let bx = coord & 7;
            let by = coord >> 3;

            debug_assert!(bx > 0 && by > 0, "this does the DC and the lower 7x7 AC");

            eob_x = cmp::max(eob_x, bx);
            eob_y = cmp::max(eob_y, by);
        }
    }

    // next step is the edge coefficients
    encode_edge::<W, ALL_PRESENT>(
        neighbors_data.left,
        neighbors_data.above,
        here,
        model_per_color,
        bool_writer,
        qt,
        pt,
        num_non_zeros_7x7,
        eob_x as u8,
        eob_y as u8,
    )
    .context(here!())?;

    // finally the DC coefficient (at 0,0)
    let predicted_val = pt.adv_predict_dc_pix::<ALL_PRESENT>(&here, qt, neighbors_data, features);

    let avg_predicted_dc = ProbabilityTables::adv_predict_or_unpredict_dc(
        here.get_dc(),
        false,
        predicted_val.predicted_dc.into(),
    );

    if here.get_dc() as i32
        != ProbabilityTables::adv_predict_or_unpredict_dc(
            avg_predicted_dc as i16,
            true,
            predicted_val.predicted_dc.into(),
        )
    {
        return err_exit_code(ExitCode::CoefficientOutOfRange, "BlockDC mismatch");
    }

    model
        .write_dc(
            bool_writer,
            pt.get_color_index(),
            avg_predicted_dc as i16,
            predicted_val.uncertainty,
            predicted_val.uncertainty2,
        )
        .context(here!())?;

    // neighbor summary is used as a predictor for the next block
    let neighbor_summary = NeighborSummary::calculate_neighbor_summary(
        &predicted_val.advanced_predict_dc_pixels_sans_dc,
        qt,
        here,
        num_non_zeros_7x7,
        features,
    );

    Ok(neighbor_summary)
}

#[inline(never)] // don't inline so that the profiler can get proper data
fn encode_edge<W: Write, const ALL_PRESENT: bool>(
    left: &AlignedBlock,
    above: &AlignedBlock,
    here: &AlignedBlock,
    model_per_color: &mut ModelPerColor,
    bool_writer: &mut VPXBoolWriter<W>,
    qt: &QuantizationTables,
    pt: &ProbabilityTables,
    num_non_zeros_7x7: u8,
    eob_x: u8,
    eob_y: u8,
) -> Result<()> {
    encode_one_edge::<W, ALL_PRESENT, true>(
        left,
        above,
        here,
        model_per_color,
        bool_writer,
        qt,
        pt,
        num_non_zeros_7x7,
        eob_x,
    )
    .context(here!())?;
    encode_one_edge::<W, ALL_PRESENT, false>(
        left,
        above,
        here,
        model_per_color,
        bool_writer,
        qt,
        pt,
        num_non_zeros_7x7,
        eob_y,
    )
    .context(here!())?;
    Ok(())
}

fn count_non_zero(v: i16) -> u8 {
    if v == 0 {
        0
    } else {
        1
    }
}

fn encode_one_edge<W: Write, const ALL_PRESENT: bool, const HORIZONTAL: bool>(
    left: &AlignedBlock,
    above: &AlignedBlock,
    block: &AlignedBlock,
    model_per_color: &mut ModelPerColor,
    bool_writer: &mut VPXBoolWriter<W>,
    qt: &QuantizationTables,
    pt: &ProbabilityTables,
    num_non_zeros_7x7: u8,
    est_eob: u8,
) -> Result<()> {
    let mut num_non_zeros_edge;

    if HORIZONTAL {
        num_non_zeros_edge = count_non_zero(block.get_coefficient(1))
            + count_non_zero(block.get_coefficient(2))
            + count_non_zero(block.get_coefficient(3))
            + count_non_zero(block.get_coefficient(4))
            + count_non_zero(block.get_coefficient(5))
            + count_non_zero(block.get_coefficient(6))
            + count_non_zero(block.get_coefficient(7));
    } else {
        num_non_zeros_edge = count_non_zero(block.get_coefficient(1 * 8))
            + count_non_zero(block.get_coefficient(2 * 8))
            + count_non_zero(block.get_coefficient(3 * 8))
            + count_non_zero(block.get_coefficient(4 * 8))
            + count_non_zero(block.get_coefficient(5 * 8))
            + count_non_zero(block.get_coefficient(6 * 8))
            + count_non_zero(block.get_coefficient(7 * 8));
    }

    model_per_color
        .write_non_zero_edge_count::<W, HORIZONTAL>(
            bool_writer,
            est_eob,
            num_non_zeros_7x7,
            num_non_zeros_edge,
        )
        .context(here!())?;

    let delta;
    let mut zig15offset;

    if HORIZONTAL {
        delta = 1;
        zig15offset = 0;
    } else {
        delta = 8;
        zig15offset = 7;
    }

    let mut coord = delta;
    for _lane in 0..7 {
        if num_non_zeros_edge == 0 {
            break;
        }

        let ptcc8 = pt.calc_coefficient_context8_lak::<ALL_PRESENT, HORIZONTAL>(
            qt,
            coord,
            &block,
            &above,
            &left,
            num_non_zeros_edge,
        );

        let coef = block.get_coefficient(coord);

        model_per_color
            .write_edge_coefficient(bool_writer, qt, coef, coord, zig15offset, &ptcc8)
            .context(here!())?;

        if coef != 0 {
            num_non_zeros_edge -= 1;
        }

        coord += delta;
        zig15offset += 1;
    }

    Ok(())
}

/// simplest case, all zeros. The goal of these test cases is go from simplest to most
/// complicated so if tests start failing, you have some idea of where to start looking.
#[test]
fn roundtrip_zeros() {
    let left = AlignedBlock::new([0; 64]);
    let above = AlignedBlock::new([0; 64]);
    let here = AlignedBlock::new([0; 64]);

    // verified output from a previous run. If this fails, then you've made a change
    // to the binary format of the compressor and probably shouldn't do that since then
    // the compressor and decompressor won't be compatible anymore.
    let verified_output = [
        0x133100e068957963,
        0x133100e068957963,
        0x133100e068957963,
        0x133100e068957963,
    ];

    roundtrip_read_write_coefficients_all(
        &left,
        &above,
        &here,
        &verified_output,
        &EnabledFeatures::compat_lepton_vector_read(),
    );
}

#[test]
fn roundtrip_ones() {
    let left = AlignedBlock::new([1; 64]);
    let above = AlignedBlock::new([1; 64]);
    let here = AlignedBlock::new([1; 64]);

    // verified output from a previous run. If this fails, then you've made a change
    // to the binary format of the compressor and probably shouldn't do that since then
    // the compressor and decompressor won't be compatible anymore.
    let verified_output = [
        0xe5a1980d71891a2b,
        0x5a8fd92548e2ea07,
        0xb392ea90d7b31238,
        0x1a769d84e98a27e,
    ];

    roundtrip_read_write_coefficients_all(
        &left,
        &above,
        &here,
        &verified_output,
        &EnabledFeatures::compat_lepton_vector_read(),
    );
}

/// test large coefficients that could overflow unpredictably if there are changes to the
/// way the math operations are performed (for example overflow or bitness)
#[test]
fn roundtrip_large_coef() {
    let left = AlignedBlock::new([1023; 64]);
    let above = AlignedBlock::new([1023; 64]);
    let here = AlignedBlock::new([1023; 64]);

    // verified output from a previous run. If this fails, then you've made a change
    // to the binary format of the compressor and probably shouldn't do that since then
    // the compressor and decompressor won't be compatible anymore.
    let verified_output = [
        0x506f55523369cf48,
        0x5b2795bc24a04d2e,
        0xdcb68ed904cfc4f9,
        0x80efab28c62db071,
    ];

    roundtrip_read_write_coefficients_all(
        &left,
        &above,
        &here,
        &verified_output,
        &EnabledFeatures::compat_lepton_scalar_read(),
    );
}

#[test]
fn roundtrip_random() {
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;

    let mut rng = StdRng::from_seed([2u8; 32]);

    let arr = [0i16; 64];

    let left = AlignedBlock::new(arr.map(|_| rng.gen_range(-1023..=1023)));
    let above = AlignedBlock::new(arr.map(|_| rng.gen_range(-1023..=1023)));
    let here = AlignedBlock::new(arr.map(|_| rng.gen_range(-1023..=1023)));

    // verified output from a previous run. If this fails, then you've made a change
    // to the binary format of the compressor and probably shouldn't do that since then
    // the compressor and decompressor won't be compatible anymore.
    let verified_output = [
        0x4b08a910feb758e8,
        0x44c3f76e93f1d204,
        0x899bc6e64957e400,
        0x322e78e37fd7ed13,
    ];

    roundtrip_read_write_coefficients_all(
        &left,
        &above,
        &here,
        &verified_output,
        &EnabledFeatures::compat_lepton_vector_read(),
    );
}

/// tests a pattern where all the coefficients are unique to make sure we don't mix up anything
#[test]
fn roundtrip_unique() {
    let mut arr = [0; 64];
    for i in 0..64 {
        arr[i] = i as i16;
    }

    let left = AlignedBlock::new(arr);
    let above = AlignedBlock::new(arr.map(|x| x + 64));
    let here = AlignedBlock::new(arr.map(|x| x + 128));

    // verified output from a previous run. If this fails, then you've made a change
    // to the binary format of the compressor and probably shouldn't do that since then
    // the compressor and decompressor won't be compatible anymore.
    let verified_output = [
        0x31caa65a4af9fe19,
        0x33622f772a9fc403,
        0xa2cc76c22f35dfbd,
        0xdb832d71fc9faf0c,
    ];

    roundtrip_read_write_coefficients_all(
        &left,
        &above,
        &here,
        &verified_output,
        &EnabledFeatures::compat_lepton_vector_read(),
    );
}

/// does all combinations of corner blocks being present or not
#[cfg(test)]
fn roundtrip_read_write_coefficients_all(
    left: &AlignedBlock,
    above: &AlignedBlock,
    here: &AlignedBlock,
    verified_output: &[u64; 4],
    features: &EnabledFeatures,
) {
    for (&(left_present, above_present), &verified_output) in
        [(false, false), (true, false), (false, true), (true, true)]
            .iter()
            .zip(verified_output.iter())
    {
        roundtrip_read_write_coefficients(
            left_present,
            above_present,
            left,
            above,
            here,
            verified_output,
            features,
        );
    }
}

/// randomizes the branches of the model so that we don't start with a
/// state where all the branches are in the same state. This is important
/// to catch any misaligment in the model state between reading and writing.
#[cfg(test)]
fn make_random_model() -> Box<Model> {
    let mut model = Model::default_boxed();

    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;

    let mut rng = StdRng::from_seed([2u8; 32]);

    model.walk(|x| {
        x.set_count(rng.gen_range(0x101..=0xffff));
    });
    model
}

/// tests the roundtrip of reading and writing coefficients
///
/// The tests are done with a seeded random model so that the tests are deterministic.
///
/// In addition, we check to make that everything ran as expected by comparing the
/// hash of the output to a verified output. This verified output is generated by
/// hashing the output plus the new state of the model.
#[cfg(test)]
fn roundtrip_read_write_coefficients(
    left_present: bool,
    above_present: bool,
    left: &AlignedBlock,
    above: &AlignedBlock,
    here: &AlignedBlock,
    verified_output: u64,
    features: &EnabledFeatures,
) {
    use crate::structs::idct::run_idct;
    use crate::structs::lepton_decoder::read_coefficient_block;
    use crate::structs::neighbor_summary::NEIGHBOR_DATA_EMPTY;
    use crate::structs::vpx_bool_reader::VPXBoolReader;

    use siphasher::sip::SipHasher13;
    use std::hash::Hasher;
    use std::io::Cursor;

    let pt = ProbabilityTables::new(0, left_present, above_present);

    let mut write_model = make_random_model();

    let mut buffer = Vec::new();

    let mut bool_writer = VPXBoolWriter::new(&mut buffer).unwrap();

    let qt = QuantizationTables::new_from_table(&[1; 64]);

    // calculate the neighbor values. Normally this is done by recycling the previous results
    // but since we are testing a one-off here, manually calculate the values
    let above_neighbor = if above_present {
        let idct_above = run_idct::<true>(above, qt.get_quantization_table_transposed());

        NeighborSummary::calculate_neighbor_summary(
            &idct_above,
            &qt,
            &above,
            above.get_count_of_non_zeros_7x7(),
            &features,
        )
    } else {
        NEIGHBOR_DATA_EMPTY
    };

    let left_neighbor = if left_present {
        let idct_left = run_idct::<true>(left, qt.get_quantization_table_transposed());

        NeighborSummary::calculate_neighbor_summary(
            &idct_left,
            &qt,
            &left,
            left.get_count_of_non_zeros_7x7(),
            &features,
        )
    } else {
        NEIGHBOR_DATA_EMPTY
    };

    let neighbors = NeighborData {
        above: &above,
        left: &left,
        above_left: &above,
        neighbor_context_above: &above_neighbor,
        neighbor_context_left: &left_neighbor,
    };

    // use the version with ALL_PRESENT is both above and left neighbors are present
    let ns_read = if left_present && above_present {
        write_coefficient_block::<true, _>(
            &pt,
            &neighbors,
            &here,
            &mut write_model,
            &mut bool_writer,
            &qt,
            &features,
        )
    } else {
        write_coefficient_block::<false, _>(
            &pt,
            &neighbors,
            &here,
            &mut write_model,
            &mut bool_writer,
            &qt,
            &features,
        )
    }
    .unwrap();

    bool_writer.finish().unwrap();

    let mut read_model = make_random_model();
    let mut bool_reader = VPXBoolReader::new(Cursor::new(&buffer)).unwrap();

    // use the version with ALL_PRESENT is both above and left neighbors are present
    let (output, ns_write) = if left_present && above_present {
        read_coefficient_block::<true, _>(
            &pt,
            &neighbors,
            &mut read_model,
            &mut bool_reader,
            &qt,
            &features,
        )
    } else {
        read_coefficient_block::<false, _>(
            &pt,
            &neighbors,
            &mut read_model,
            &mut bool_reader,
            &qt,
            &features,
        )
    }
    .unwrap();

    assert_eq!(ns_write.get_num_non_zeros(), ns_read.get_num_non_zeros());
    assert_eq!(ns_write.get_horizontal(), ns_read.get_horizontal());
    assert_eq!(ns_write.get_vertical(), ns_read.get_vertical());
    assert_eq!(output.get_block(), here.get_block());
    assert_eq!(write_model.model_checksum(), read_model.model_checksum());

    let mut h = SipHasher13::new();
    h.write(&buffer);
    h.write_u64(write_model.model_checksum());
    let hash = h.finish();

    println!("0x{:x?},", hash);

    if verified_output != 0 {
        assert_eq!(
            verified_output, hash,
            "Hash mismatch. Unexpected change in model behavior/output format"
        );
    }
}
