/***PROPRIETARY_STATEMENT********************************************
 *
 * IBM CONFIDENTIAL
 *
 * OCO SOURCE MATERIALS
 *
 * 5650-ZOS
 *
 * COPYRIGHT IBM CORP. 2020
 *
 * THE SOURCE CODE FOR THIS PROGRAM IS NOT PUBLISHED OR OTHERWISE
 * DIVESTED OF ITS TRADE SECRETS, IRRESPECTIVE OF WHAT HAS BEEN
 * DEPOSITED WITH THE U.S. COPYRIGHT OFFICE.
 *
 * STATUS = HBB77D0
 *
 ********************************************************************
 *
 */

#ifndef CONVERT_H_
#define CONVERT_H_

#include <inttypes.h>
#include <stddef.h>

// Functions to convert data format.
uint64_t fp16_to_dlf16(uint16_t *input_fp16_data, uint16_t *output_dflt16_data,
    uint64_t nbr_fields_to_convert);
uint64_t fp32_to_dlf16(
    float *input_data, uint16_t *output_data, uint64_t nbr_fields_to_convert);
uint64_t bfloat_to_dlf16(uint16_t *input_data, uint16_t *output_data,
    uint64_t nbr_fields_to_convert);
uint64_t dlf16_to_fp16(uint16_t *input_dflt16_data, uint16_t *output_fp16_data,
    uint64_t nbr_fields_to_convert);
uint64_t dlf16_to_fp32(
    uint16_t *input_data, float *output_data, uint64_t nbr_fields_to_convert);
uint64_t dlf16_to_bfloat(uint16_t *input_data, uint16_t *output_data,
    uint64_t nbr_fields_to_convert);
#endif /* CONVERT_H_ */
