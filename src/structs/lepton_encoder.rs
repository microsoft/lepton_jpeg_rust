/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use anyhow::{Context, Result};
use bytemuck::cast;
use wide::{i16x8, i32x8, CmpEq};

use std::io::Write;

use crate::consts::UNZIGZAG_49_TR;
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
        neighbor_summary_component.resize(num_non_zeros_length, NeighborSummary::default());

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
    here_tr: &AlignedBlock,
    model: &mut Model,
    bool_writer: &mut VPXBoolWriter<W>,
    qt: &QuantizationTables,
    features: &EnabledFeatures,
) -> Result<NeighborSummary> {
    let model_per_color = model.get_per_color(pt);

    // using SIMD instructions, construct a 64 bit mask of all
    // the non-zero coefficients in the block, cmp_eq returns 0xffff for zero coefficients
    let block_simd: [i16x8; 8] = cast(*here_tr.get_block());

    let mut nonzero_mask = 0;
    for i in 0..8 {
        nonzero_mask |= (block_simd[i].cmp_eq(i16x8::ZERO).move_mask() as u64) << (8 * i);
    }
    nonzero_mask = !nonzero_mask;

    let mask_7x7 = nonzero_mask & 0xFEFEFEFEFEFEFE00;

    // First we encode the 49 inner coefficients

    // calculate the predictor context bin based on the neighbors
    let num_non_zeros_7x7_context_bin =
        pt.calc_num_non_zeros_7x7_context_bin::<ALL_PRESENT>(neighbors_data);

    // store how many of these coefficients are non-zero, which is used both
    // to terminate the loop early and as a predictor for the model
    let num_non_zeros_7x7 = mask_7x7.count_ones() as u8;

    model_per_color
        .write_non_zero_7x7_count(
            bool_writer,
            num_non_zeros_7x7_context_bin,
            num_non_zeros_7x7,
        )
        .context(here!())?;

    let mut num_non_zeros_7x7_remaining = num_non_zeros_7x7 as usize;

    if num_non_zeros_7x7_remaining > 0 {
        let best_priors = pt.calc_coefficient_context_7x7_aavg_block::<ALL_PRESENT>(
            neighbors_data.left,
            neighbors_data.above,
            neighbors_data.above_left,
        );
        // calculate the bin we are using for the number of non-zeros
        let mut num_non_zeros_remaining_bin =
            ProbabilityTables::num_non_zeros_to_bin_7x7(num_non_zeros_7x7_remaining);

        // now loop through the coefficients in zigzag, terminating once we hit the number of non-zeros
        for (zig49, &coord) in UNZIGZAG_49_TR.iter().enumerate() {
            let best_prior_bit_length = u16_bit_length(best_priors[coord as usize]);

            let coef = here_tr.get_coefficient(coord as usize);

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

    // Next step is the edge coefficients.
    // Here we produce the first part of edge DCT coefficients predictions for neighborhood blocks
    // and transposed raster of dequantized DCT coefficients with 0 in DC
    let (raster, horiz_pred, vert_pred) = encode_edge::<W, ALL_PRESENT>(
        neighbors_data,
        &here_tr,
        model_per_color,
        bool_writer,
        qt,
        pt,
        nonzero_mask,
    )
    .context(here!())?;

    // finally the DC coefficient (at 0,0)
    let q0 = qt.get_quantization_table()[0] as i32;
    let predicted_val =
        pt.adv_predict_dc_pix::<ALL_PRESENT>(&raster, q0, &neighbors_data, features);

    let avg_predicted_dc = ProbabilityTables::adv_predict_or_unpredict_dc(
        here_tr.get_dc(),
        false,
        predicted_val.predicted_dc,
    );

    if here_tr.get_dc() as i32
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
    let neighbor_summary = NeighborSummary::new(
        &predicted_val.advanced_predict_dc_pixels_sans_dc,
        here_tr.get_dc() as i32 * q0,
        num_non_zeros_7x7,
        horiz_pred,
        vert_pred,
        features,
    );

    Ok(neighbor_summary)
}

//#[inline(never)] // don't inline so that the profiler can get proper data
fn encode_edge<W: Write, const ALL_PRESENT: bool>(
    neighbors_data: &NeighborData,
    here_tr: &AlignedBlock,
    model_per_color: &mut ModelPerColor,
    bool_writer: &mut VPXBoolWriter<W>,
    qt: &QuantizationTables,
    pt: &ProbabilityTables,
    nonzero_mask: u64,
) -> Result<([i32x8; 8], i32x8, i32x8)> {
    let q_tr: AlignedBlock = AlignedBlock::new(cast(*qt.get_quantization_table_transposed()));

    let raster: [i32x8; 8] = [
        (q_tr.as_i16x8(0) & i16x8::new([0, -1, -1, -1, -1, -1, -1, -1]))
            .mul_widen(here_tr.as_i16x8(0)),
        q_tr.as_i16x8(1).mul_widen(here_tr.as_i16x8(1)),
        q_tr.as_i16x8(2).mul_widen(here_tr.as_i16x8(2)),
        q_tr.as_i16x8(3).mul_widen(here_tr.as_i16x8(3)),
        q_tr.as_i16x8(4).mul_widen(here_tr.as_i16x8(4)),
        q_tr.as_i16x8(5).mul_widen(here_tr.as_i16x8(5)),
        q_tr.as_i16x8(6).mul_widen(here_tr.as_i16x8(6)),
        q_tr.as_i16x8(7).mul_widen(here_tr.as_i16x8(7)),
    ];

    let (h_pred, v_pred) =
        ProbabilityTables::predict_current_edges(neighbors_data, nonzero_mask, &raster);

    // here we calculate the furthest x and y coordinates that have non-zero coefficients
    // which are used as predictors for the number of edge coefficients
    let mask_7x7 = (nonzero_mask & 0xFEFEFEFEFEFEFE00) | 1;
    let mut mask_y = mask_7x7 | (mask_7x7 << 32);
    mask_y |= mask_y << 16;
    mask_y |= mask_y << 8;

    // effectively (7 - eob) of DB Lepton
    let eob_y: u8 = mask_y.leading_zeros() as u8;
    let eob_x: u8 = (mask_7x7.leading_zeros() >> 3) as u8;

    let num_non_zeros_bin = (mask_7x7.count_ones() as u8 + 2) / 7;

    let num_non_zeros_edge_x = (nonzero_mask & 0x0101010101010100).count_ones() as u8;
    encode_one_edge::<W, ALL_PRESENT, true>(
        here_tr,
        model_per_color,
        bool_writer,
        &h_pred.to_array(),
        qt,
        pt,
        num_non_zeros_bin,
        eob_x,
        num_non_zeros_edge_x,
    )
    .context(here!())?;

    let num_non_zeros_edge_y = (nonzero_mask & 0xFE).count_ones() as u8;
    encode_one_edge::<W, ALL_PRESENT, false>(
        here_tr,
        model_per_color,
        bool_writer,
        &v_pred.to_array(),
        qt,
        pt,
        num_non_zeros_bin,
        eob_y,
        num_non_zeros_edge_y,
    )
    .context(here!())?;

    let (h_pred, v_pred) = ProbabilityTables::predict_next_edges(&raster, nonzero_mask);
    Ok((raster, h_pred, v_pred))
}

fn encode_one_edge<W: Write, const ALL_PRESENT: bool, const HORIZONTAL: bool>(
    block: &AlignedBlock,
    model_per_color: &mut ModelPerColor,
    bool_writer: &mut VPXBoolWriter<W>,
    pred: &[i32; 8],
    qt: &QuantizationTables,
    pt: &ProbabilityTables,
    num_non_zeros_bin: u8,
    est_eob: u8,
    num_non_zeros_edge: u8,
) -> Result<()> {
    model_per_color
        .write_non_zero_edge_count::<W, HORIZONTAL>(
            bool_writer,
            est_eob,
            num_non_zeros_bin,
            num_non_zeros_edge,
        )
        .context(here!())?;

    let delta;
    let mut zig15offset;

    if HORIZONTAL {
        delta = 8;
        zig15offset = 0;
    } else {
        delta = 1;
        zig15offset = 7;
    }

    let mut coord_tr = delta;
    let mut num_non_zeros_left = num_non_zeros_edge;

    for _lane in 0..7 {
        if num_non_zeros_left == 0 {
            break;
        }

        let best_prior =
            pt.calc_coefficient_context8_lak::<ALL_PRESENT, HORIZONTAL>(qt, coord_tr, pred);

        let coef = block.get_coefficient(coord_tr);

        model_per_color
            .write_edge_coefficient(
                bool_writer,
                qt,
                coef,
                zig15offset,
                num_non_zeros_left,
                best_prior,
            )
            .context(here!())?;

        if coef != 0 {
            num_non_zeros_left -= 1;
        }

        coord_tr += delta;
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
        0x7AC1898CC7C24813,
        0x7AC1898CC7C24813,
        0x7AC1898CC7C24813,
        0x7AC1898CC7C24813,
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
        0x9BFB8B62B8F5C8C9,
        0x93AF2386E2B30C18,
        0x108F59190036D45A,
        0xD7A3F6449681E97F,
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
        0x3195FA3A5611FE57,
        0xB438BE8570092738,
        0xD6441DDBA177AC7E,
        0x15A42C2C0A53C332,
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
        0x05F9A5949CB1517C,
        0x006A051435067451,
        0x42715401410F1602,
        0x394329CA90795246,
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
        0xEB20783DD8B86A4E,
        0xFF351314979E8CC6,
        0xC07C557B9318D5D0,
        0x38C278C1A08C66B4,
    ];

    roundtrip_read_write_coefficients_all(
        &left,
        &above,
        &here,
        &verified_output,
        &EnabledFeatures::compat_lepton_vector_read(),
    );
}

/// tests a pattern to check the non-zero counting
#[test]
fn roundtrip_non_zeros_counts() {
    let mut arr = [0; 64];

    // upper left corner is all 50, the rest is 0
    for i in 0..64 {
        arr[i] = if (i & 7) < 4 || (i >> 3) < 4 { 50 } else { 0 };
    }

    let block = AlignedBlock::new(arr);

    // verified output from a previous run. If this fails, then you've made a change
    // to the binary format of the compressor and probably shouldn't do that since then
    // the compressor and decompressor won't be compatible anymore.
    let verified_output = [
        0xBB28316BA25CE552,
        0x3BF2227D37C68152,
        0xB78F8A203D96A9E6,
        0x627DB637DA12F0F0,
    ];

    roundtrip_read_write_coefficients_all(
        &block,
        &block,
        &block,
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
    use crate::structs::idct::get_i32x8;
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
    let q = AlignedBlock::new(cast(*qt.get_quantization_table_transposed()));

    // all the work is done in transposed raster coefficients order
    let here_block = AlignedBlock::new(cast(i16x8::transpose(cast(*here.get_block()))));
    let above_block = AlignedBlock::new(cast(i16x8::transpose(cast(*above.get_block()))));
    let left_block = AlignedBlock::new(cast(i16x8::transpose(cast(*left.get_block()))));

    // calculate the neighbor values. Normally this is done by recycling the previous results
    // but since we are testing a one-off here, manually calculate the values
    let calculate_summary = |block: &AlignedBlock| {
        let mut raster: [i32x8; 8] = [0.into(); 8]; // transposed

        for col in 0..8 {
            raster[col] = get_i32x8(col, &block) * get_i32x8(col, &q);
        }
        raster[0].as_array_mut()[0] = 0; // DC coefficient is always 0 since it is calculated last

        let (h_pred, v_pred) = ProbabilityTables::predict_next_edges(&mut raster, u64::MAX);

        let idct_above = run_idct(&raster);

        NeighborSummary::new(
            &idct_above,
            block.get_dc() as i32 * q.get_coefficient(0) as i32,
            block.get_count_of_non_zeros_7x7(),
            h_pred,
            v_pred,
            &features,
        )
    };

    let above_neighbor = if above_present {
        calculate_summary(&above_block)
    } else {
        NEIGHBOR_DATA_EMPTY
    };

    let left_neighbor = if left_present {
        calculate_summary(&left_block)
    } else {
        NEIGHBOR_DATA_EMPTY
    };

    let neighbors = NeighborData {
        above: &above_block,
        left: &left_block,
        above_left: &above_block,
        neighbor_context_above: &above_neighbor,
        neighbor_context_left: &left_neighbor,
    };

    // use the version with ALL_PRESENT is both above and left neighbors are present
    let ns_read = if left_present && above_present {
        write_coefficient_block::<true, _>(
            &pt,
            &neighbors,
            &here_block,
            &mut write_model,
            &mut bool_writer,
            &qt,
            &features,
        )
    } else {
        write_coefficient_block::<false, _>(
            &pt,
            &neighbors,
            &here_block,
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
    assert_eq!(output.get_block(), here_block.get_block());
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
