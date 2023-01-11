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

/**
 * <h2>ONNX-MLIR Runtime Java API documentation</h2>
 *
 * <h3>Introduction</h3>
 *
 * ONNX-MLIR project comes with an executable `onnx-mlir` capable
 * of compiling onnx models to a jar. In this documentation, we
 * demonstrate how to interact programmatically with the compiled
 * jar using ONNX-MLIR's Java Runtime API.
 *
 * <h4>Java Runtime API</h4>
 *
 * <h4>Classes</h4>
 *
 * `OMModel` is the class implementing the default model entry point
 * and input/output signature functions.
 *
 * `OMTensor` is the class used to describe the runtime information
 * (rank, shape, data type, etc) associated with a tensor input or
 * output.
 *
 * `OMTensorList` is the class used to hold an array of OMTensor so
 * that they can be passed into and out of the compiled model as inputs
 * and outputs.
 *
 * <h4>Model Entry Point Signature</h4>
 *
 * All compiled models will have the same exact Java function signature
 * equivalent to:
 *
 * ```java
 * public static OMTensorList mainGraph(OMTensorList list)
 * ```
 *
 * Intuitively, the model takes a list of tensors as input and returns
 * a list of tensors as output.
 *
 * <h4>Invoke Models Using Java Runtime API</h4>
 *
 * We demonstrate using the API functions to run a simple ONNX model
 * consisting of an add operation. To create such an onnx model, use
 * this <a href="gen_add_onnx.py" target="_blank"><b>python script</b></a>
 *
 * To compile the above model, run `onnx-mlir --EmitJNI add.onnx` and a jar
 * file "add.jar" should appear. We can use the following Java code to call
 * into the compiled function computing the sum of two inputs:
 *
 * ```java
 * import com.ibm.onnxmlir.OMModel;
 * import com.ibm.onnxmlir.OMTensor;
 * import com.ibm.onnxmlir.OMTensorList;
 *
 * public class Add {
 *   public static void main(String[] args) {
 *     // Shared shape.
 *     long[] shape = {3, 2};
 *     // Construct x1 omt filled with 1.
 *     float[] x1Data = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
 *     OMTensor x1 = new OMTensor(x1Data, shape);
 *     // Construct x2 omt filled with 2.
 *     float[] x2Data = {2.f, 2.f, 2.f, 2.f, 2.f, 2.f};
 *     OMTensor x2 = new OMTensor(x2Data, shape);
 *     // Construct a list of omts as input.
 *     OMTensor[] list = new OMTensor[]{x1, x2};
 *     OMTensorList input = new OMTensorList(list);
 *     // Call the compiled onnx model function.
 *     OMTensorList output = OMModel.mainGraph(input);
 *     // Get the first omt as output.
 *     OMTensor[] y = output.getOmtArray();
 *     float[] outputPtr = y[0].getFloatData();
 *     // Print its content, should be all 3.
 *     for (int i = 0; i < 6; i++)
 *       System.out.print(outputPtr[i] + " ");
 *   }
 * }
 * ```
 *
 * Compile with `javac -cp javaruntime.jar Add.java`, you should
 * see a class `Add.class` appearing. Run it with `java -cp .:add.jar Add`,
 * and the output should be:
 *
 * ```
 * 3.0 3.0 3.0 3.0 3.0 3.0
 * ```
 * Exactly as it should be.
 */
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
			      PosixFilePermissions.fromString("rwx------"));

			// Load the temporary .so copy
			System.load(lib.toString());
			logger.finer(lib.toString() + " loaded");

		    } catch (IOException e2) {
			logger.severe(e2.getMessage());
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
            logger.severe(e.getMessage());
        }
    }

    private static native OMTensorList main_graph_jni(OMTensorList list);
    private static native String[] query_entry_points();
    private static native String input_signature_jni(String entry_point);
    private static native String output_signature_jni(String entry_point);

    /**
     * Default model runtime entry point
     *
     * @param list input tensor list
     * @return output tensor list
     */
    public static OMTensorList mainGraph(OMTensorList list) {
        return main_graph_jni(list);
    }

    /**
     * Query all entry point names in the model.
     *
     * @return String array of entry point names
     */
    public static String[] queryEntryPoints() {
	return query_entry_points();
    }

    /**
     * Input signature of default model runtime entry point
     *
     * @return JSON string of input signature
     */
    public static String inputSignature() {
        return input_signature_jni("run_main_graph");
    }

    public static String inputSignature(String entry_point) {
        return input_signature_jni(entry_point);
    }

    /**
     * Output signature of default model runtime entry point
     *
     * @return JSON string of output signature
     */
    public static String outputSignature() {
        return output_signature_jni("run_main_graph");
    }

    public static String outputSignature(String entry_point) {
        return output_signature_jni(entry_point);
    }
}
