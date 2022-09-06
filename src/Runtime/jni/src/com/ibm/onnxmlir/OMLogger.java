// SPDX-License-Identifier: Apache-2.0

package com.ibm.onnxmlir;

import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashMap;
import java.util.logging.ConsoleHandler;
import java.util.logging.FileHandler;
import java.util.logging.Formatter;
import java.util.logging.Level;
import java.util.logging.LogRecord;
import java.util.logging.Logger;

// Subclass to implement our own formatter
class OMFormatter extends Formatter {
    private StackTraceElement getCallerStackFrame(final String callerName) {
        final StackTraceElement stack[] = new Throwable().getStackTrace();
        StackTraceElement callerFrame = null;

        // Search the stack trace to find the calling class
        for (int i = 0; i < stack.length; i++) {
            final StackTraceElement frame = stack[i];
            if (callerName.equals(frame.getClassName())) {
                callerFrame = frame;
                break;
            }
        }

        return callerFrame;
    }

    // Override the format method with out own
    @Override
    public String format(LogRecord lr) {
	final String FMT = "yyyy-MM-dd HH:mm:ss Z";
	final StackTraceElement cf = getCallerStackFrame(lr.getSourceClassName());

        return String.format("[%1$s][%2$s][%3$s]%4$s:%5$s:%6$s %7$s\n",
                             new SimpleDateFormat(FMT).format(new Date(lr.getMillis())),
			     Thread.currentThread().getName(),
                             lr.getLevel().getName(),
			     cf.getFileName(),
			     lr.getSourceMethodName(),
			     cf.getLineNumber(),
			     formatMessage(lr));
    }
}

public class OMLogger {
    static HashMap<String, Level> map = new HashMap<>();

    // Mapping between C and Java log levels
    static {
        map.put("trace",   Level.FINEST);
        map.put("debug",   Level.FINER);
        map.put("info",    Level.INFO);
        map.put("warning", Level.WARNING);
        map.put("error",   Level.SEVERE);
        map.put("fatal",   Level.SEVERE); // SEVERE is the highest Java level
    }

    public static void initLogger(Logger logger) {
        // Set logger level
        String level = System.getenv("ONNX_MLIR_JNI_LOG_LEVEL");
        Level l = map.get(level == null ? "info" : level);
        logger.setLevel(l == null ? Level.INFO : l);

        // Set logger handler
        String file = System.getenv("ONNX_MLIR_JNI_LOG_FILE");
        if (file == null || file.equals("stdout") || file.equals("stderr")) {
            ConsoleHandler ch = new ConsoleHandler();
	    ch.setFormatter(new OMFormatter());
            ch.setLevel(l == null ? Level.INFO : l); // Must also set handler level
            logger.addHandler(ch);
        }
        else {
            try {
                FileHandler fh =
		    new FileHandler(file + "." + Thread.currentThread().getName());
		fh.setFormatter(new OMFormatter());
		fh.setLevel(l == null ? Level.INFO : l); // Must also set handler level
                logger.addHandler(fh);
            } catch (SecurityException|IOException e) {
                System.out.println(e.getMessage());
            }
        }
    }
}
