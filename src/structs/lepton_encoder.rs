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
    quantization_tables::QuantizationTables, row_spec::RowSpec, truncate_components::*,
    vpx_bool_writer::VPXBoolWriter,
};

use default_boxed::DefaultBoxed;

use super::block_context::NeighborData;
use super::probability_tables_set::PTS;

#[inline(never)] // don't inline so that the profiler can get proper data
pub fn lepton_encode_row_range<W: Write>(
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
        let component = cur_row.component;

        let left_model;
        let middle_model;

        if is_top_row[component] {
            is_top_row[component] = false;

            left_model = &PTS.corner[component];
            middle_model = &PTS.top[component];
        } else {
            left_model = &PTS.mid_left[component];
            middle_model = &PTS.middle[component];
        }

        process_row(
            &mut model,
            &mut bool_writer,
            left_model,
            middle_model,
            &image_data[component],
            &quantization_tables[component],
            &mut neighbor_summary_cache[component][..],
            cur_row.curr_y,
            component_size_in_blocks[component],
            features,
        )
        .context(here!())?;
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
    left_model: &ProbabilityTables,
    middle_model: &ProbabilityTables,
    image_data: &BlockBasedImage,
    qt: &QuantizationTables,
    neighbor_summary_cache: &mut [NeighborSummary],
    curr_y: i32,
    component_size_in_block: i32,
    features: &EnabledFeatures,
) -> Result<()> {
    let mut block_context = image_data.off_y(curr_y);
    let block_width = image_data.get_block_width();

    for jpeg_x in 0..block_width {
        let pt: &ProbabilityTables = if jpeg_x == 0 {
            left_model
        } else {
            middle_model
        };

        // shortcut all the checks for the presence of left/right components by passing a constant generic parameter
        if pt.is_all_present() {
            serialize_tokens::<W, true>(
                &block_context,
                qt,
                pt,
                model,
                image_data,
                neighbor_summary_cache,
                bool_writer,
                features,
            )
            .context(here!())?;
        } else {
            serialize_tokens::<W, false>(
                &block_context,
                qt,
                pt,
                model,
                image_data,
                neighbor_summary_cache,
                bool_writer,
                features,
            )
            .context(here!())?;
        }

        let offset = block_context.next();

        if offset >= component_size_in_block {
            return Ok(());
        }
    }

    Ok(())
}

#[inline(never)] // don't inline so that the profiler can get proper data
fn serialize_tokens<W: Write, const ALL_PRESENT: bool>(
    context: &BlockContext,
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

    // First we encode the 49 inner coefficients

    // calculate the predictor context bin based on the neighbors
    let num_non_zeros_7x7_context_bin =
        pt.calc_num_non_zeros_7x7_context_bin::<ALL_PRESENT>(neighbors_data);

    // store how many of these coefficients are non-zero, which is used both
    // to terminate the loop early and as a predictor for the model
    let num_non_zeros_7x7 = here.get_count_of_non_zeros_7x7();

    model_per_color
        .write_non_zero_7x7_count(
            bool_writer,
            num_non_zeros_7x7_context_bin,
            num_non_zeros_7x7,
        )
        .context(here!())?;

    // these are used as predictors for the number of non-zero edge coefficients
    // do math in 32 bits since this is faster on most modern platforms
    let mut eob_x: u32 = 0;
    let mut eob_y: u32 = 0;

    let mut num_non_zeros_7x7_remaining = num_non_zeros_7x7 as usize;

    let best_priors = pt.calc_coefficient_context_7x7_aavg_block::<ALL_PRESENT>(
        neighbors_data.left,
        neighbors_data.above,
        neighbors_data.above_left,
    );

    if num_non_zeros_7x7_remaining > 0 {
        // calculate the bin we are using for the number of non-zeros
        let mut num_non_zeros_remaining_bin =
            ProbabilityTables::num_non_zeros_to_bin_7x7(num_non_zeros_7x7_remaining);

        // now loop through the coefficients in zigzag, terminating once we hit the number of non-zeros
        for (zig49, &coord) in UNZIGZAG_49.iter().enumerate() {
            let best_prior_bit_length = u16_bit_length(best_priors[coord as usize]);

            let coef = here.get_coefficient(coord as usize);

            model_per_color
                .write_coef(
                    bool_writer,
                    coef,
                    zig49,
                    num_non_zeros_remaining_bin,
                    best_prior_bit_length as usize,
                )
                .context(here!())?;

            if coef != 0 {
                // here we calculate the furthest x and y coordinates that have non-zero coefficients
                // which is later used as a predictor for the number of edge coefficients
                let bx = u32::from(coord) & 7;
                let by = u32::from(coord) >> 3;

                debug_assert!(bx > 0 && by > 0, "this does the DC and the lower 7x7 AC");

                eob_x = cmp::max(eob_x, bx);
                eob_y = cmp::max(eob_y, by);

                num_non_zeros_7x7_remaining -= 1;
                if num_non_zeros_7x7_remaining == 0 {
                    break;
                }

                // update the bin since the number of non-zeros has changed
                num_non_zeros_remaining_bin =
                    ProbabilityTables::num_non_zeros_to_bin_7x7(num_non_zeros_7x7_remaining);
            }
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
        predicted_val.predicted_dc,
    );

    if here.get_dc() as i32
        != ProbabilityTables::adv_predict_or_unpredict_dc(
            avg_predicted_dc as i16,
            true,
            predicted_val.predicted_dc,
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

        let best_prior = pt.calc_coefficient_context8_lak::<ALL_PRESENT, HORIZONTAL>(
            qt, coord, &block, &above, &left,
        );

        let coef = block.get_coefficient(coord);

        model_per_color
            .write_edge_coefficient(
                bool_writer,
                qt,
                coef,
                coord,
                zig15offset,
                num_non_zeros_edge,
                best_prior,
            )
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
    let block = AlignedBlock::new([0; 64]);

    roundtrip_read_write_coefficients(
        &block,
        &block,
        &block,
        &block,
        [1; 64],
        0xd7c55f4988eaf7d5,
        &EnabledFeatures::compat_lepton_vector_read(),
    );
}

/// tests blocks with only DC coefficient set
#[test]
fn roundtrip_dc_only() {
    let mut block = AlignedBlock::new([0; 64]);
    block.set_dc(-100);

    roundtrip_read_write_coefficients(
        &block,
        &block,
        &block,
        &block,
        [1; 64],
        0x2dcc28548ce40dec,
        &EnabledFeatures::compat_lepton_vector_read(),
    );
}

/// tests blocks with only edge coefficients set
#[test]
fn roundtrip_edges_only() {
    let mut block = AlignedBlock::new([0; 64]);
    for i in 1..7 {
        block.set_coefficient(i, -100);
        block.set_coefficient(i * 8, 100);
    }

    roundtrip_read_write_coefficients(
        &block,
        &block,
        &block,
        &block,
        [1; 64],
        0x60cb33137d9ba75f,
        &EnabledFeatures::compat_lepton_vector_read(),
    );
}

/// tests blocks with only 7x7 coefficients set
#[test]
fn roundtrip_ac_only() {
    let mut block = AlignedBlock::new([0; 64]);
    for i in 0..64 {
        let x = i & 7;
        let y = i >> 3;

        if x > 0 && y > 0 {
            block.set_coefficient(i, (x * y) as i16);
        }
    }

    roundtrip_read_write_coefficients(
        &block,
        &block,
        &block,
        &block,
        [1; 64],
        0x782acca7e2ee50a3,
        &EnabledFeatures::compat_lepton_vector_read(),
    );
}

#[test]
fn roundtrip_ones() {
    let block = AlignedBlock::new([1; 64]);

    roundtrip_read_write_coefficients(
        &block,
        &block,
        &block,
        &block,
        [1; 64],
        0xd986b8703f95c0fd,
        &EnabledFeatures::compat_lepton_vector_read(),
    );
}

/// test large coefficients that could overflow unpredictably if there are changes to the
/// way the math operations are performed (for example overflow or bitness)
#[test]
fn roundtrip_large_coef() {
    // largest coefficient that doesn't cause a DC overflow
    let block = AlignedBlock::new([-1010; 64]);

    roundtrip_read_write_coefficients(
        &block,
        &block,
        &block,
        &block,
        [1; 64],
        0x9e97dff50bc0188,
        &EnabledFeatures::compat_lepton_vector_read(),
    );

    // now test with maximum quantization table. In theory this is legal according
    // the JPEG format and there is no code preventing this from being attempted
    // by the encoder.

    roundtrip_read_write_coefficients(
        &block,
        &block,
        &block,
        &block,
        [65535; 64],
        0xcac19d7e86aece1b,
        &EnabledFeatures::compat_lepton_vector_read(),
    );
}

/// "random" set of blocks to ensure that all ranges of coefficients work properly
#[test]
fn roundtrip_random_seed() {
    use rand::Rng;

    // the 127 seed is a choice that doesn't overflow the DC coefficient
    // since the encoder is somewhat picky if the DC estimate overflows
    // it also has different behavior for 32 and 16 bit codepath
    let mut rng = crate::helpers::get_rand_from_seed([127; 32]);

    let arr = [0i16; 64];

    let left = AlignedBlock::new(arr.map(|_| rng.gen_range(-2047..=2047)));
    let above = AlignedBlock::new(arr.map(|_| rng.gen_range(-2047..=2047)));
    let here = AlignedBlock::new(arr.map(|_| rng.gen_range(-2047..=2047)));
    let above_left = AlignedBlock::new(arr.map(|_| rng.gen_range(-2047..=2047)));
    let qt = arr.map(|_| rng.gen_range(1u16..=65535));

    // using 32 bit math (test emulating both scalar and vector C++ code)
    let a = roundtrip_read_write_coefficients(
        &left,
        &above,
        &above_left,
        &here,
        qt,
        0xe3c687262f0df4f5,
        &EnabledFeatures::compat_lepton_scalar_read(),
    );

    // using 16 bit math
    let b = roundtrip_read_write_coefficients(
        &left,
        &above,
        &above_left,
        &here,
        qt,
        0xdbacb31b714489fc,
        &EnabledFeatures::compat_lepton_vector_read(),
    );

    assert!(a != b);
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
    let above_left = AlignedBlock::new(arr.map(|x| x + 128));
    let here = AlignedBlock::new(arr.map(|x| x + 256));

    roundtrip_read_write_coefficients(
        &left,
        &above,
        &above_left,
        &here,
        [1; 64],
        0x36f907a4d7f80559,
        &EnabledFeatures::compat_lepton_vector_read(),
    );
}

/// tests a pattern to check the non-zero counting
#[test]
fn roundtrip_non_zeros_counts() {
    let mut arr = [0; 64];

    // upper left corner is all 50, the rest is 0
    // this should result in 3 or 4 non-zero coefficients
    // (depending on vertical/horizontal, make this non-symetrical to catch mixups)
    for i in 0..64 {
        let x = i & 7;
        let y = i >> 3;

        arr[i] = if x < 4 && y < 3 { 50 } else { 0 };
    }

    let block = AlignedBlock::new(arr);

    roundtrip_read_write_coefficients(
        &block,
        &block,
        &block,
        &block,
        [1; 64],
        0xb4031bacdb0c911b,
        &EnabledFeatures::compat_lepton_vector_read(),
    );
}

/// randomizes the branches of the model so that we don't start with a
/// state where all the branches are in the same state. This is important
/// to catch any misaligment in the model state between reading and writing.
#[cfg(test)]
fn make_random_model() -> Box<Model> {
    let mut model = Model::default_boxed();

    use rand::Rng;

    let mut rng = crate::helpers::get_rand_from_seed([2u8; 32]);

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
    left: &AlignedBlock,
    above: &AlignedBlock,
    above_left: &AlignedBlock,
    here: &AlignedBlock,
    qt: [u16; 64],
    verified_output: u64,
    features: &EnabledFeatures,
) -> u64 {
    use crate::structs::{
        block_based_image::EMPTY_BLOCK, lepton_decoder::read_coefficient_block,
        neighbor_summary::NEIGHBOR_DATA_EMPTY, vpx_bool_reader::VPXBoolReader,
    };

    // use the Sip hasher directly since that's guaranteed not to change implementation vs the default hasher
    use siphasher::sip::SipHasher13;
    use std::hash::Hasher;
    use std::io::{Cursor, Read};

    let mut write_model = make_random_model();

    let mut buffer = Vec::new();

    let mut bool_writer = VPXBoolWriter::new(&mut buffer).unwrap();

    let qt = QuantizationTables::new_from_table(&qt);

    /// This is a helper function to avoid having to duplicate the code for the different cases.
    fn call_write_coefficient_block<W: Write>(
        left: Option<(&AlignedBlock, &NeighborSummary)>,
        above: Option<(&AlignedBlock, &NeighborSummary)>,
        above_left: Option<&AlignedBlock>,
        here: &AlignedBlock,
        write_model: &mut Model,
        bool_writer: &mut VPXBoolWriter<W>,
        qt: &QuantizationTables,
        features: &EnabledFeatures,
    ) -> NeighborSummary {
        let pt = ProbabilityTables::new(0, left.is_some(), above.is_some());
        let n = NeighborData {
            above: above.map(|x| x.0).unwrap_or(&EMPTY_BLOCK),
            left: left.map(|x| x.0).unwrap_or(&EMPTY_BLOCK),
            above_left: above_left.unwrap_or(&EMPTY_BLOCK),
            neighbor_context_above: above.map(|x| x.1).unwrap_or(&NEIGHBOR_DATA_EMPTY),
            neighbor_context_left: left.map(|x| x.1).unwrap_or(&NEIGHBOR_DATA_EMPTY),
        };

        // call the right version depending on if we have all neighbors or not
        if left.is_some() && above.is_some() {
            write_coefficient_block::<true, _>(
                &pt,
                &n,
                &here,
                write_model,
                bool_writer,
                qt,
                features,
            )
            .unwrap()
        } else {
            write_coefficient_block::<false, _>(
                &pt,
                &n,
                &here,
                write_model,
                bool_writer,
                qt,
                features,
            )
            .unwrap()
        }
    }

    /// This is a helper function to avoid having to duplicate the code for the different cases.
    fn call_read_coefficient_block<R: Read>(
        left: Option<(&AlignedBlock, &NeighborSummary)>,
        above: Option<(&AlignedBlock, &NeighborSummary)>,
        above_left: Option<&AlignedBlock>,
        read_model: &mut Model,
        bool_reader: &mut VPXBoolReader<R>,
        qt: &QuantizationTables,
        features: &EnabledFeatures,
    ) -> (AlignedBlock, NeighborSummary) {
        let pt = ProbabilityTables::new(0, left.is_some(), above.is_some());
        let n = NeighborData {
            above: above.map(|x| x.0).unwrap_or(&EMPTY_BLOCK),
            left: left.map(|x| x.0).unwrap_or(&EMPTY_BLOCK),
            above_left: above_left.unwrap_or(&EMPTY_BLOCK),
            neighbor_context_above: above.map(|x| x.1).unwrap_or(&NEIGHBOR_DATA_EMPTY),
            neighbor_context_left: left.map(|x| x.1).unwrap_or(&NEIGHBOR_DATA_EMPTY),
        };

        // call the right version depending on if we have all neighbors or not
        if left.is_some() && above.is_some() {
            read_coefficient_block::<true, _>(&pt, &n, read_model, bool_reader, qt, features)
                .unwrap()
        } else {
            read_coefficient_block::<false, _>(&pt, &n, read_model, bool_reader, qt, features)
                .unwrap()
        }
    }

    // overall idea here is to call write and read on all possible permutations of neighbors
    // the grid looks like this:
    //
    // [ above_left ] [ above ]
    // [ left       ] [ here  ]
    //
    // first: above_left (with no neighbors)

    let w_above_left_ns = call_write_coefficient_block(
        None,
        None,
        None,
        &above_left,
        &mut write_model,
        &mut bool_writer,
        &qt,
        &features,
    );

    // now above, with above_left as neighbor
    let w_above_ns = call_write_coefficient_block(
        Some((&above_left, &w_above_left_ns)),
        None,
        None,
        &above,
        &mut write_model,
        &mut bool_writer,
        &qt,
        &features,
    );

    // now left with above_left as neighbor
    let w_left_ns = call_write_coefficient_block(
        None,
        Some((&above_left, &w_above_left_ns)),
        None,
        &left,
        &mut write_model,
        &mut bool_writer,
        &qt,
        &features,
    );

    // now here with above and left as neighbors
    let w_here_ns = call_write_coefficient_block(
        Some((&left, &w_left_ns)),
        Some((&above, &w_above_ns)),
        Some(above_left),
        &here,
        &mut write_model,
        &mut bool_writer,
        &qt,
        &features,
    );

    bool_writer.finish().unwrap();

    // now re-read the model and make sure everything matches
    let mut read_model = make_random_model();
    let mut bool_reader = VPXBoolReader::new(Cursor::new(&buffer)).unwrap();

    let (r_above_left_block, r_above_left_ns) = call_read_coefficient_block(
        None,
        None,
        None,
        &mut read_model,
        &mut bool_reader,
        &qt,
        &features,
    );

    assert_eq!(r_above_left_block.get_block(), above_left.get_block());
    assert_eq!(r_above_left_ns, w_above_left_ns);

    let (r_above_block, r_above_ns) = call_read_coefficient_block(
        Some((&r_above_left_block, &w_above_left_ns)),
        None,
        None,
        &mut read_model,
        &mut bool_reader,
        &qt,
        &features,
    );

    assert_eq!(r_above_block.get_block(), above.get_block());
    assert_eq!(r_above_ns, w_above_ns);

    let (r_left_block, r_left_ns) = call_read_coefficient_block(
        None,
        Some((&r_above_left_block, &r_above_left_ns)),
        None,
        &mut read_model,
        &mut bool_reader,
        &qt,
        &features,
    );

    assert_eq!(r_left_block.get_block(), left.get_block());
    assert_eq!(r_left_ns, w_left_ns);

    let (r_here, r_here_ns) = call_read_coefficient_block(
        Some((&r_left_block, &r_left_ns)),
        Some((&r_above_block, &r_above_ns)),
        Some(above_left),
        &mut read_model,
        &mut bool_reader,
        &qt,
        &features,
    );

    assert_eq!(r_here.get_block(), here.get_block());
    assert_eq!(r_here_ns, w_here_ns);

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

    hash
}
