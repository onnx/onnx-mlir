// SPDX-License-Identifier: Apache-2.0

package com.ibm.onnxmlir;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.ShortBuffer;
import java.util.ArrayList;
import java.util.Base64;
import java.util.HashMap;

import com.jsoniter.JsonIterator;
import com.jsoniter.output.JsonStream;

/*import com.fasterxml.jackson.databind.ObjectMapper;*/

/* This class is used to run the backend JNI tests.
 *
 * It reads from stdin a JSON string which contains the
 * input tensors converted from numpy ndarray. The tensor
 * data are base64 encoded.
 *
 * It decodes the base64 tensor data, reconstruct the
 * OMTensor and OMTensorList objects from the JSON, and
 * calls the OMModel.mainGraph entry point of the model.
 *
 * With the output tensors returned by the OMModel.mainGraph
 * entry point of the model, it base64 encodes the tensor
 * data, and construct a HashMap with the OMTensor and
 * OMTensorList objects.
 *
 * It writes to stdout a JSON string which contains the
 * output tensors converted from Java HashMap. The tensor
 * data are base64 encoded.
 */
public class OMRunner
{
    private static class Data {
	/* For Jsoniter, field names match the keys in the JSON */
	String buffer;
	String dtype;
	long[] shape;

	/* For Jackson, words after removing get/set of getter/setter
	 * methods with the first letter lowercased match the keys in
	 * the JSON.
	 */
	/*
	String getBuffer() { return this.buffer; }
	void setBuffer(String buffer) { this.buffer = buffer; }

	String getDtype() { return this.dtype; }
	void setDtype(String dtype) { this.dtype = dtype; }

	long[] getShape() { return this.shape; }
	void setShape(long[] shape) { this.shape = shape; }
	*/
    }

    private static final HashMap<String, ByteOrder> numpy2javaEndian =
	new HashMap<String, ByteOrder>() {{
	    put(">", ByteOrder.BIG_ENDIAN);
	    put("<", ByteOrder.LITTLE_ENDIAN);
	    put("=", ByteOrder.nativeOrder());
	    put("|", ByteOrder.nativeOrder());
	}};

    private static final HashMap<ByteOrder, String> java2numpyEndian =
	new HashMap<ByteOrder, String>() {{
	    put(ByteOrder.BIG_ENDIAN,    ">");;
	    put(ByteOrder.LITTLE_ENDIAN, "<");;
	}};
    private static final String numpyEndian =
	java2numpyEndian.get(ByteOrder.nativeOrder());

    private static final HashMap<String, Integer> numpy2onnxType =
	new HashMap<String, Integer>() {{
	    put("b1", OMTensor.ONNX_TYPE_BOOL);
	    put("i1", OMTensor.ONNX_TYPE_INT8);
	    put("u1", OMTensor.ONNX_TYPE_UINT8);
	    put("i2", OMTensor.ONNX_TYPE_INT16);
	    put("u2", OMTensor.ONNX_TYPE_UINT16);
	    put("i4", OMTensor.ONNX_TYPE_INT32);
	    put("u4", OMTensor.ONNX_TYPE_UINT32);
	    put("i8", OMTensor.ONNX_TYPE_INT64);
	    put("u8", OMTensor.ONNX_TYPE_UINT64);
	    put("f2", OMTensor.ONNX_TYPE_FLOAT16);
	    put("f4", OMTensor.ONNX_TYPE_FLOAT);
	    put("f8", OMTensor.ONNX_TYPE_DOUBLE);
	}};

    private static final HashMap<Integer, String> onnx2numpyType =
	new HashMap<Integer, String>() {{
	    put(OMTensor.ONNX_TYPE_BOOL,               "|b1");
	    put(OMTensor.ONNX_TYPE_INT8,               "|i1");
	    put(OMTensor.ONNX_TYPE_UINT8,              "|u1");
	    put(OMTensor.ONNX_TYPE_INT16,   numpyEndian+"i2");
	    put(OMTensor.ONNX_TYPE_UINT16,  numpyEndian+"u2");
	    put(OMTensor.ONNX_TYPE_INT32,   numpyEndian+"i4");
	    put(OMTensor.ONNX_TYPE_UINT32,  numpyEndian+"u4");
	    put(OMTensor.ONNX_TYPE_INT64,   numpyEndian+"i8");
	    put(OMTensor.ONNX_TYPE_UINT64,  numpyEndian+"u8");
	    put(OMTensor.ONNX_TYPE_FLOAT16, numpyEndian+"f2");
	    put(OMTensor.ONNX_TYPE_FLOAT,   numpyEndian+"f4");
	    put(OMTensor.ONNX_TYPE_DOUBLE,  numpyEndian+"f8");
	}};

    private static OMTensor createTensor(String buffer, long[] shape, String dtype) {
	/* We need a ByteBuffer for OMTensor but ByteBuffer.wrap(bytes)
	 * does NOT work. Because wrap simply creates a "view" of the
	 * byte[] as ByteBuffer. The backing byte[] is a Java object
	 * but the JNI wrapper is expecting a real direct ByteBuffer
	 * to hold the data to be given to the native code.
	 */
	byte[] bytes = Base64.getDecoder().decode(buffer);
	ByteBuffer data = ByteBuffer.allocateDirect(bytes.length);
        data.put(bytes);

	String e = dtype.substring(0, 1);
	ByteOrder endian = numpy2javaEndian.get(e);

	String t = dtype.substring(1);
	Integer otype = numpy2onnxType.get(t);

	return new OMTensor(data, shape, endian,
			    otype == null ? -1 : otype.intValue());
    }

    private static HashMap<String, Object> encodeTensor(OMTensor omt) throws Exception {
	/* We need a byte[] for base64 encode but buffer.array()
	 * does NOT work. Because the buffer is backed by JNI code
	 * generated array, not a Java byte[] object. So base64
	 * encode results in UnsupportedOperationException.
	 */
	ByteBuffer buffer = omt.getData();
	byte[] bytes = new byte[buffer.limit()];
	buffer.get(bytes);

	String dtype = onnx2numpyType.get(omt.getDataType());

	HashMap<String, Object> map = new HashMap<String, Object>();
	map.put("buffer", Base64.getEncoder().encodeToString(bytes/*buffer.array()*/));
	map.put("dtype", dtype);
	map.put("shape", omt.getShape());
	return map;
    }

    /* Model inputs are read from stdin encoded in JSON. This routine will
     *
     * - read JSON from stdin
     * - decode JSON array with Jsoniter
     * - call createTensor to create an OMTensor from each object
     * - construct the OMTensorList to be fed into mainGraph
     */
    private static OMTensorList readStdin() throws Exception {
        BufferedReader stdin =
	    new BufferedReader(new InputStreamReader(System.in));
	ArrayList<OMTensor> omtl = new ArrayList<OMTensor>();

	JsonIterator json = JsonIterator.parse(stdin.readLine());
	int count = 0;
	while(json.readArray()) {
	    Data data = json.read(Data.class);
	    OMTensor omt = createTensor(data.buffer, data.shape, data.dtype);
	    omtl.add(omt);
	    count++;
	}
	OMTensor[] omts = new OMTensor[count];
	return new OMTensorList(omtl.toArray(omts));
    }

    /* Model inputs are read from stdin encoded in JSON. This routine will
     *
     * - read JSON from stdin
     * - decode JSON array with Jackson
     * - call createTensor to create an OMTensor from each object
     * - construct the OMTensorList to be fed into mainGraph
     */
    /*
    private static OMTensorList readStdin2() throws Exception {
	ObjectMapper om = new ObjectMapper();
	Data[] data = om.readValue(System.in, Data[].class);
	OMTensor[] omts = new OMTensor[data.length];
	for (int i = 0; i < data.length; i++) {
	    omts[i] = createTensor(data[i].buffer, data[i].shape, data[i].dtype);
	}
	return new OMTensorList(omts);
    }
    */

    /* Model outputs are written to stdout encoded in JSON. This routine will
     *
     * - loop through tensors in the OMTensorList returned from mainGraph
     * - call encodeTensor to create a list of HashMap from each OMTensor
     * - encode HashMap list into JSON with Jsoniter
     * - write JSON to stdout
     */
    private static void writeStdout(OMTensorList output) throws Exception {
	ArrayList<HashMap<String, Object>> list = new ArrayList<HashMap<String, Object>>();
	HashMap<String, Object> map = new HashMap<String, Object>();
	OMTensor[] omts = output.getOmtArray();

	for (int i = 0; i < omts.length; i++) {
	    list.add(encodeTensor(omts[i]));
	}

	BufferedWriter stdout =
	    new BufferedWriter(new OutputStreamWriter(System.out));
	stdout.write(JsonStream.serialize(list));
	stdout.flush();
    }

    /* Model outputs are written to stdout encoded in JSON. This routine will
     *
     * - loop through tensors in the OMTensorList returned from mainGraph
     * - call encodeTensor to create a list of HashMap from each OMTensor
     * - encode HashMap list into JSON with Jackson
     * - write JSON to stdout
     */
    /*
    private static void writeStdout2(OMTensorList output) throws Exception {
	ArrayList<HashMap<String, Object>> list = new ArrayList<HashMap<String, Object>>();
	HashMap<String, Object> map = new HashMap<String, Object>();
	OMTensor[] omts = output.getOmtArray();

	for (int i = 0; i < omts.length; i++) {
	    list.add(encodeTensor(omts[i]));
	}

	ObjectMapper om = new ObjectMapper();
	om.writeValue(System.out, list);
    }
    */

    /* Read inputs from stdin, call mainGraph, write outputs to stdout */
    public static void main(String[] args) throws Exception {
	writeStdout(OMModel.mainGraph(readStdin()));
    }
}
