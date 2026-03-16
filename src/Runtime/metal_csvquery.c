/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-- metal_csvquery.c - Metal C CSVQUERY Implementation ---------------===//
//
// Copyright 2026 IBM Corporation
//
// =============================================================================
//
// This file contains Metal C implementation for querying module information
// using the z/OS CSVQUERY system service.
//
//===----------------------------------------------------------------------===//

#ifndef __IBM_METAL__
  #error "xlc -qMETAL compile option required."
#endif

#pragma longName  /* Allow names longer than 8 chars */
#pragma prolog(metal_get_module_path, main)
#pragma epilog(metal_get_module_path, main)

// Forward declaration with OS linkage
#pragma linkage(metal_get_module_path, OS)

/// Get the pathname of the module containing a given address.
///
/// This function uses the z/OS CSVQUERY system service to retrieve the
/// pathname of the load module that contains the specified address.
/// When called with the function's own address, it returns the pathname
/// of the currently executing module (executable or shared library).
///
/// \param[out] output_buffer Buffer to receive the pathname string
/// \param[in]  buffer_size   Size of the output buffer in bytes
///
/// \return Length of the pathname on success, 0 on error
///
/// This function is thread-safe.
///
int metal_get_module_path(char* output_buffer, int buffer_size) {
    int CSV_Retcode = 0;
    
    // CSVQUERY OUTPATHNAME format: 2-byte big-endian length + 1024-byte pathname
    char filename[1026] = {0};
    
    // Use this function's address to query its own module
    void* my_address = (void*)metal_get_module_path;
    
    // Declare the list form of the CSVQUERY parameter list
    __asm ( " CSVQUERY PLISTVER=IMPLIED_VERSION,"
                   "MF=(L,CSV_CSVQUERY) " : "DS"(CSV_CSVQUERY) );
    
    // Execute CSVQUERY to get the pathname of the module containing my_address
    // INADDR64: Input 64-bit address within the module to query
    // OUTPATHNAME: Output pathname (2-byte length + pathname data)
    // RETCODE: Return code (0 = success)
    // MF=(E,...): Execute form using the parameter list declared above
    __asm ( " CSVQUERY INADDR64=%3,"
         "OUTPATHNAME=%0,"
         "RETCODE=%1,"
         "MF=(E,%2) "
       : "=m"(*filename),
         "=m"(CSV_Retcode),
         "=m"(CSV_CSVQUERY)
       : "m"(my_address) : "r0","r1","r14","r15");
    
    // Process the result if CSVQUERY succeeded
    if (CSV_Retcode == 0 && output_buffer != 0) {
        // Extract pathname length from first 2 bytes (big-endian format)
        unsigned short pathname_len = ((unsigned char)filename[0] << 8) |
                                      ((unsigned char)filename[1]);
        
        if (pathname_len > 0) {
            int i;
            int max_copy = pathname_len;
            
            // Prevent buffer overflow
            if (max_copy >= buffer_size) {
                max_copy = buffer_size - 1;
            }
            
            // Copy pathname from offset 2 (after the length field)
            for (i = 0; i < max_copy; i++) {
                output_buffer[i] = filename[i + 2];
            }
            output_buffer[max_copy] = 0;  // Null terminate
            
            return pathname_len;  // Return actual pathname length
        }
    }
    
    return 0;  // Error or no pathname available
}
