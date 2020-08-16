package com.ibm.onnxmlir;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.jar.JarFile;

public class DynEntryPoint {
    static String libname = "libmodel.so";

    static {
        File jar;
        JarFile jf;
        String jarDir = null;
        String libPath = null;
        try {
            // Get path name of jar
            jar = new File(DynEntryPoint.class.getProtectionDomain()
                                              .getCodeSource()
                                              .getLocation().toURI());
            jarDir = jar.getParentFile().getAbsolutePath();
            libPath = jarDir + "/" + libname;

            // Open jar file to read and check libname inside jar.
            // If IOException thrown, load .so from where .jar is.
            //
            // Checking whether DynEntryPoint.class.getResourceAsStream returns null
            // does NOT work. Because it checks whether the resource is
            // available on the classpath, not only just inside the jar file.
            jf = new JarFile(jar);
            if (jf.getEntry(libname) != null) {
                File libFile = new File(libPath);
                // Copy .so to where jar is
                Files.copy(jf.getInputStream(jf.getEntry(libname)),
                        libFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
                // z/OS USS requires "x" permission bit
                libFile.setExecutable(true, false);
                // Load the temporary .so copy
                System.load(libPath);
                // POSIX can unlink file after loading
                libFile.delete();
            }
            else {
                // Throw subclass of IOException
                throw new FileNotFoundException(".so not found inside jar");
            }
        } catch (URISyntaxException e) {
            // Failed to find jar path, assume the .so is in cwd
            System.load(libname);
        } catch (IOException e) {
            // .so not found in jar, assume it's where jar is
            System.load(libPath);
        }
    }

    private static native RtMemRefList main_graph_jni(RtMemRefList ormrd);
    
    public static RtMemRefList main_graph(RtMemRefList ormrd) {
        return main_graph_jni(ormrd);
    }
}
