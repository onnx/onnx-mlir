# SPDX-License-Identifier: Apache-2.0

import os
import re
import struct


def get_version_from_so(so_path):
    """Extract onnx-mlir version from a shared library."""
    if not os.path.isfile(so_path):
        raise FileNotFoundError(f"Shared library not found: {so_path}")

    with open(so_path, "rb") as f:
        # Read ELF header
        elf_header = f.read(64)
        if elf_header[:4] != b"\x7fELF":
            raise ValueError("Not a valid ELF file")

        # Determine 32-bit or 64-bit
        is_64bit = elf_header[4] == 2
        endian = "<" if elf_header[5] == 1 else ">"

        # Get section header offset and size
        if is_64bit:
            shoff = struct.unpack(endian + "Q", elf_header[40:48])[0]
            shentsize = struct.unpack(endian + "H", elf_header[58:60])[0]
            shnum = struct.unpack(endian + "H", elf_header[60:62])[0]
        else:
            shoff = struct.unpack(endian + "I", elf_header[32:36])[0]
            shentsize = struct.unpack(endian + "H", elf_header[46:48])[0]
            shnum = struct.unpack(endian + "H", elf_header[48:50])[0]

        # Read section headers
        f.seek(shoff)
        sections = []
        for _ in range(shnum):
            sections.append(f.read(shentsize))

        # Find .comment section
        comment_data = None
        for sh in sections:
            # Skip to sh_name field (offset 0 for both 32 and 64-bit)
            sh_name_off = struct.unpack(endian + "I", sh[0:4])[0]
            # We need to read string table to get name, but for now just check type
            sh_type = struct.unpack(endian + "I", sh[4:8])[0]
            if sh_type == 1:  # SHT_PROGBITS
                # Check if it's .comment by looking at flags and addr
                sh_flags = struct.unpack(endian + "Q" if is_64bit else "I",
                                         sh[8:16] if is_64bit else sh[8:12])[0]
                sh_addr = struct.unpack(endian + "Q" if is_64bit else "I",
                                      sh[16:24] if is_64bit else sh[12:16])[0]
                if sh_flags == 0 and sh_addr == 0:  # .comment section characteristics
                    sh_offset = struct.unpack(endian + "Q" if is_64bit else "I",
                                            sh[24:32] if is_64bit else sh[16:20])[0]
                    sh_size = struct.unpack(endian + "Q" if is_64bit else "I",
                                          sh[32:40] if is_64bit else sh[20:24])[0]
                    f.seek(sh_offset)
                    comment_data = f.read(sh_size)
                    break

        if comment_data is None:
            return None

        # Extract version string
        match = re.search(rb"onnx-mlir\s+(\S+)", comment_data)
        if match:
            return match.group(1).decode("utf-8")
        return None