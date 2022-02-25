// SPDX-License-Identifier: Apache-2.0

package com.ibm.onnxmlir;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.nio.file.attribute.PosixFilePermissions;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import java.util.logging.Logger;

public class OMModel {
    private static final Logger logger = Logger.getLogger(OMModel.class.getName());

    static {
	OMLogger.initLogger(logger);

        try {
            // Get path of jar
	    Path jar = Paths.get(OMModel.class.getProtectionDomain()
                                              .getCodeSource()
                                              .getLocation().toURI());
	    Path jarDir = jar.getParent();

	    // All .so files found will be copied under the ./native subdirectory
	    // where the jar file resides.
	    Path tmpDir = Files.createTempDirectory(jarDir, "native_");

            // Open jar file to load all .so libs inside. The .so is loaded
            // by OS so it must be extracted to the file system.
            JarFile jf = new JarFile(jar.toFile());
	    for (Enumeration<JarEntry> e = jf.entries(); e.hasMoreElements(); ) {
		String libFile = e.nextElement().getName();

		if (libFile.endsWith(".so")) {
		    try {
			Path lib = Paths.get(tmpDir.toString(), libFile);
			Path libDir = lib.getParent();

			Files.createDirectories(libDir);

			// Copy .so to the temporary directory
			Files.copy(jf.getInputStream(jf.getEntry(libFile)), lib,
				   StandardCopyOption.REPLACE_EXISTING);

			// z/OS USS requires "x" permission bit
			Files.setPosixFilePermissions(lib,
			      PosixFilePermissions.fromString("rwxr-xr-x"));

			// Load the temporary .so copy
			System.load(lib.toString());
			logger.finer(lib.toString() + " loaded");

		    } catch (IOException e2) {
			e2.printStackTrace();
		    }
		} // if
            } // for

	    // POSIX can unlink the .so once they have been loaded so
	    // remove ./native and all subdirectories. Note it's important
	    // to sort in reverse order so we delete all the files under
	    // a subdirectory first before deleting the subdirectory itself.
	    // Otherwise, the subdirectory may fail to be deleted since
	    // it may not be empty.
	    Files.walk(tmpDir)
                 .sorted(Comparator.reverseOrder())
                 .peek(p -> logger.finer(p.toString()))
                 .map(p -> p.toFile())
                 .forEach(f -> f.delete());

        } catch (URISyntaxException|IOException e) {
            e.printStackTrace();
        }
    }

    private static native OMTensorList main_graph_jni(OMTensorList list);
    private static native String input_signature_jni();
    private static native String input_signature_jni(String entry_point);
    private static native String output_signature_jni();
    private static native String output_signature_jni(String entry_point);
    
    public static OMTensorList mainGraph(OMTensorList list) {
        return main_graph_jni(list);
    }

    public static String inputSignature() {
        return input_signature_jni("run_main_graph");
    }

    public static String inputSignature(String entry_point) {
        return input_signature_jni(entry_point);
    }

    public static String outputSignature() {
        return output_signature_jni("run_main_graph");
    }

    public static String outputSignature(String entry_point) {
        return output_signature_jni(entry_point);
    }
}
