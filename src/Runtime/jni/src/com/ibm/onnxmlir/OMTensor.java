// SPDX-License-Identifier: Apache-2.0

package com.ibm.onnxmlir;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.ShortBuffer;

/**
 * Class describing the runtime information such as rank,
 * shape, strides, data type, etc. associated with a tensor
 * input/output.
 */
public class OMTensor {

    /* We can use enum but that creates another class
     * which complicates things for JNI.
     *
     * Values are standard ONNX data types defined in
     * https://github.com/onnx/onnx/blob/main/onnx/onnx.proto#L484
     */
    public final static int ONNX_TYPE_UNDEFINED  = 0;
    public final static int ONNX_TYPE_FLOAT      = 1;
    public final static int ONNX_TYPE_UINT8      = 2;
    public final static int ONNX_TYPE_INT8       = 3;
    public final static int ONNX_TYPE_UINT16     = 4;
    public final static int ONNX_TYPE_INT16      = 5;
    public final static int ONNX_TYPE_INT32      = 6;
    public final static int ONNX_TYPE_INT64      = 7;
    public final static int ONNX_TYPE_STRING     = 8;
    public final static int ONNX_TYPE_BOOL       = 9;
    public final static int ONNX_TYPE_FLOAT16    = 10;
    public final static int ONNX_TYPE_DOUBLE     = 11;
    public final static int ONNX_TYPE_UINT32     = 12;
    public final static int ONNX_TYPE_UINT64     = 13;
    public final static int ONNX_TYPE_COMPLEX64  = 14;
    public final static int ONNX_TYPE_COMPLEX128 = 15;
    public final static int ONNX_TYPE_BFLOAT16   = 16;
    public final static int ONNX_TYPE_FLOAT8E4M3FN = 17;
    public final static int ONNX_TYPE_FLOAT8E4M3FNUZ = 18;
    public final static int ONNX_TYPE_FLOAT8E5M2 = 19;
    public final static int ONNX_TYPE_FLOAT8E5M2FNUZ = 20;
    public final static int LAST_ONNX_TYPE = 20; // Alias for last ONNX_TYPE.

    public final static int[] ONNX_TYPE_SIZE = new int[] {
            0,  /* UNDEFINED  */
            4,  /* FLOAT      */
            1,  /* UINT8      */
            1,  /* INT8       */
            2,  /* UINT16     */
            2,  /* INT16      */
            4,  /* INT32      */
            8,  /* INT64      */
            0,  /* STRING     */
            1,  /* BOOL       */
            2,  /* FLOAT16    */
            8,  /* DOUBLE     */
            4,  /* UINT32     */
            8,  /* UINT64     */
            8,  /* COMPLEX64  */
            16, /* COMPLEX128 */
            2,  /* BFLOAT16   */
            1,  /* FLOAT8E4M3FN */
            1,  /* FLOAT8E4M3FNUZ */
            1,  /* FLOAT8E5M2 */
            1,  /* FLOAT8E5M2FNUZ */
    };

    public final static String[] ONNX_TYPE_NAME = new String[] {
            "UNDEFINED",
            "FLOAT",
            "UINT8",
            "INT8",
            "UINT16",
            "INT16",
            "INT32",
            "INT64",
            "STRING",
            "BOOL",
            "FLOAT16",
            "DOUBLE",
            "UINT32",
            "UINT64",
            "COMPLEX64",
            "COMPLEX128",
            "BFLOAT16",
            "FLOAT8E4M3FN",
            "FLOAT8E4M3FNUZ",
            "FLOAT8E5M2",
            "FLOAT8E5M2FNUZ",
    };

    private final ByteOrder nativeEndian = ByteOrder.nativeOrder();

    private ByteBuffer _data;
    private long[] _shape;
    private long[] _strides;
    private int _dataType;
    /* Use int internally since rank is the length of shape[]
     * and Java array length is int.
     */
    private int _rank;

    /**
     * Constructor
     *
     * @param data byte data array for tensor
     * @param shape data shape array
     * @param flag true for boolean tensor, false for byte tensor
     *
     * @return OMTensor with boolean or byte data
     */
    public OMTensor(byte[] data, long[] shape, boolean flag) {
	if (flag)
	    setBoolData(data);
	else
	    setByteData(data);
	putShape(shape);
    }

    /**
     * Constructor
     *
     * @param data byte data array for tensor
     * @param shape data shape array
     *
     * @return OMTensor with byte data
     */
    public OMTensor(byte[] data, long[] shape) {
	this(data, shape, false);
    }

    /**
     * Constructor
     *
     * @param data short data array for tensor
     * @param shape data shape array
     *
     * @return OMTensor with short data
     */
    public OMTensor(short[] data, long[] shape) {
        setShortData(data);
        putShape(shape);
    }

    /**
     * Constructor
     *
     * @param data int data array for tensor
     * @param shape data shape array
     *
     * @return OMTensor with int data
     */
    public OMTensor(int[] data, long[] shape) {
        setIntData(data);
        putShape(shape);
    }

    /**
     * Constructor
     *
     * @param data long data array for tensor
     * @param shape data shape array
     *
     * @return OMTensor with long data
     */
    public OMTensor(long[] data, long[] shape) {
        setLongData(data);
        putShape(shape);
    }

    /**
     * Constructor
     *
     * @param data float data array for tensor
     * @param shape data shape array
     *
     * @return OMTensor with float data
     */
    public OMTensor(float[] data, long[] shape) {
        setFloatData(data);
        putShape(shape);
    }

    /**
     * Constructor
     *
     * @param data double data array for tensor
     * @param shape data shape array
     *
     * @return OMTensor with double data
     */
    public OMTensor(double[] data, long[] shape) {
        setDoubleData(data);
        putShape(shape);
    }


    /* ---------- Bool data getter and setter ---------- */
    /* On the native side bool is backed by byte so bool */
    /* array is the same as byte array.                  */

    /**
     * Bool data getter
     *
     * @return bool data array
     */
    public byte[] getBoolData() {
	if (_dataType != ONNX_TYPE_BOOL)
	    throw new NumberFormatException("Data type is " +
					    ONNX_TYPE_NAME[_dataType]);
	if (_data == null) return null;

        /* asReadOnlyBuffer() creates a new view so the position of the
         * original data will stay at 0 for subsequent getByteData()
         * after get(b).
         */
        byte[] b = new byte[_data.limit()];
        _data.asReadOnlyBuffer().get(b);
        return b;
    }

    /**
     * Bool data setter
     *
     * @param data bool array to be set
     */
    public void setBoolData(byte[] data) {
        /* slice() creates a new view so the position of the
         * original data will stay at 0 for getByteData() after put(data).
         */
        _data = ByteBuffer.allocateDirect(data.length);
        _data.slice().put(data);
        _dataType = ONNX_TYPE_INT8;
    }

    /* ---------- Byte data getter and setter ---------- */

    /**
     * Byte data getter
     *
     * @return byte data array
     */
    public byte[] getByteData() {
	if (_dataType != ONNX_TYPE_INT8 && _dataType != ONNX_TYPE_UINT8)
	    throw new NumberFormatException("Data type is " +
					    ONNX_TYPE_NAME[_dataType]);
	if (_data == null) return null;

        /* asReadOnlyBuffer() creates a new view so the position of the
         * original data will stay at 0 for subsequent getByteData()
         * after get(b).
         */
        byte[] b = new byte[_data.limit()];
        _data.asReadOnlyBuffer().get(b);
        return b;
    }

    /**
     * Byte data setter
     *
     * @param data byte array to be set
     */
    public void setByteData(byte[] data) {
        /* slice() creates a new view so the position of the
         * original data will stay at 0 for getByteData() after put(data).
         */
        _data = ByteBuffer.allocateDirect(data.length);
        _data.slice().put(data);
        _dataType = ONNX_TYPE_INT8;
    }

    /* ---------- Short data getter and setter ---------- */

    /**
     * Short data getter
     *
     * @return short data array
     */
    public short[] getShortData() {
	if (_dataType != ONNX_TYPE_INT16 && _dataType != ONNX_TYPE_UINT16)
	    throw new NumberFormatException("Data type is " +
					    ONNX_TYPE_NAME[_dataType]);
        if (_data == null) return null;

        /* asShortBuffer() creates a new view so the position of the
         * original data will stay at 0 for subsequent getShortData()
         * after get(s).
         */
        ShortBuffer sb = _data.asShortBuffer();
        short[] s = new short[sb.limit()];
        sb.get(s);
        return s;
    }

    /**
     * Short data setter
     *
     * @param data short array to be set
     */
    public void setShortData(short[] data) {
        /* asShortBuffer() creates a new view so the position of the
         * original data will stay at 0 for getShortData() after put(data).
         */
        _data = ByteBuffer.allocateDirect(data.length*2).order(nativeEndian);
        _data.asShortBuffer().put(data);
        _dataType = ONNX_TYPE_INT16;
    }

    /* ---------- Int data getter and setter ---------- */

    /**
     * Int data getter
     *
     * @return int data array
     */
    public int[] getIntData() {
	if (_dataType != ONNX_TYPE_INT32 && _dataType != ONNX_TYPE_UINT32)
	    throw new NumberFormatException("Data type is " +
					    ONNX_TYPE_NAME[_dataType]);
        if (_data == null) return null;

        /* asIntBuffer() creates a new view so the position of the
         * original data will stay at 0 for subsequent getIntData()
         * after get(i).
         */
        IntBuffer ib = _data.asIntBuffer();
        int[] i = new int[ib.limit()];
        ib.get(i);
        return i;
    }

    /**
     * Int data setter
     *
     * @param data int array to be set
     */
    public void setIntData(int[] data) {
        /* asIntBuffer() creates a new view so the position of the
         * original data will stay at 0 for getIntData() after put(data).
         */
        _data = ByteBuffer.allocateDirect(data.length*4).order(nativeEndian);
        _data.asIntBuffer().put(data);
        _dataType = ONNX_TYPE_INT32;
    }

    /* ---------- Long data getter and setter ---------- */

    /**
     * Long data getter
     *
     * @return long data array
     */
    public long[] getLongData() {
	if (_dataType != ONNX_TYPE_INT64 && _dataType != ONNX_TYPE_UINT64)
	    throw new NumberFormatException("Data type is " +
					    ONNX_TYPE_NAME[_dataType]);
        if (_data == null) return null;

        /* asLongBuffer() creates a new view so the position of the
         * original data will stay at 0 for subsequent getLongData()
         * after get(l).
         */
        LongBuffer lb = _data.asLongBuffer();
        long[] l = new long[lb.limit()];
        lb.get(l);
        return l;
    }

    /**
     * Long data setter
     *
     * @param data long array to be set
     */
    public void setLongData(long[] data) {
        /* asLongBuffer() creates a new view so the position of the
         * original data will stay at 0 for getLongData() after put(data).
         */
        _data = ByteBuffer.allocateDirect(data.length*8).order(nativeEndian);
        _data.asLongBuffer().put(data);
        _dataType = ONNX_TYPE_INT64;
    }

    /* ---------- Float data getter and setter ---------- */

    /**
     * Float data getter
     *
     * @return float data array
     */
    public float[] getFloatData() {
	if (_dataType != ONNX_TYPE_FLOAT)
	    throw new NumberFormatException("Data type is " +
					    ONNX_TYPE_NAME[_dataType]);
        if (_data == null) return null;

        /* asFloatBuffer() creates a new view so the position of the
         * original data will stay at 0 for subsequent getFloatData()
         * after get(f).
         */
        FloatBuffer fb = _data.asFloatBuffer();
        float[] f = new float[fb.limit()];
        fb.get(f);
        return f;
    }

    /**
     * Float data setter
     *
     * @param data float array to be set
     */
    public void setFloatData(float[] data) {
        /* asFloatBuffer() creates a new view so the position of the
         * original data will stay at 0 for getFloatData() after put(data).
         */
        _data = ByteBuffer.allocateDirect(data.length*4).order(nativeEndian);
        _data.asFloatBuffer().put(data);
        _dataType = ONNX_TYPE_FLOAT;
    }

    /* ---------- Double data getter and setter ---------- */

    /**
     * Double data getter
     *
     * @return double data array
     */
    public double[] getDoubleData() {
	if (_dataType != ONNX_TYPE_DOUBLE)
	    throw new NumberFormatException("Data type is " +
					    ONNX_TYPE_NAME[_dataType]);
        if (_data == null) return null;

        /* asDoubleBuffer() creates a new view so the position of the
         * original data will stay at 0 for subsequent getDoubleData()
         * after get(d).
         */
        DoubleBuffer db = _data.asDoubleBuffer();
        double[] d = new double[db.limit()];
        db.get(d);
        return d;
    }

    /**
     * Double data setter
     *
     * @param data double array to be set
     */
    public void setDoubleData(double[] data) {
        /* asDoubleBuffer() creates a new view so the position of the
         * original data will stay at 0 for getDoubleData() after put(data).
         */
        _data = ByteBuffer.allocateDirect(data.length*8).order(nativeEndian);
        _data.asDoubleBuffer().put(data);
        _dataType = ONNX_TYPE_DOUBLE;
    }

    /* ---------- Data shape getter and setter ---------- */

    /**
     * Data shape getter
     *
     * @return data shape array
     */
    public long[] getShape() {
        return _shape;
    }

    /**
     * Data shape setter
     *
     * @param shape data shape array to be set
     */
    public void setShape(long[] shape) {
        if (shape.length != _rank)
            throw new IllegalArgumentException(
                    "array length " + shape.length + " != rank " + _rank);
        _shape = shape;
    }

    /* ---------- Data strides getter and setter ---------- */

    /**
     * Data strides getter
     *
     * @return data strides array
     */
    public long[] getStrides() {
        return _strides;
    }

    /**
     * Data strides setter
     *
     * @param strides data strides array to be set
     */
    public void setStrides(long[] strides) {
        if (strides.length != _rank)
            throw new IllegalArgumentException(
                    "array length " + strides.length + " != rank " + _rank);
        _strides = strides;
    }

    /* ---------- Data type getter and setter ---------- */

    /**
     * Data type getter
     *
     * @return data type
     */
    public int getDataType() {
        return _dataType;
    }

    /**
     * Data type setter
     *
     * @param dataType data type to be set
     */
    public void setDataType(int dataType) {
        if (dataType < 0 || dataType > LAST_ONNX_TYPE)
            throw new IllegalArgumentException(
                    "data type " + dataType + " unknown");
        _dataType = dataType;
    }

    /* ---------- Data buffer size getter ---------- */

    /**
     * Data buffer size getter
     *
     * @return total size of the data buffer in bytes
     */
    public long getBufferSize() {
        return _data == null ? 0 : _data.limit();
    }

    /* ---------- Rank getter ---------- */

    /**
     * Rank getter (return long to be consistent with C/C++ API)
     *
     * @return rank of the OMTensor
     */
    public long getRank() {
        return (long)_rank;
    }

    /* ---------- Number of elements getter ---------- */

    /**
     * Number of elements getter
     *
     * @return number of data elements in the data buffer
     */
    public long getNumElems() {
        if (_shape.length == 0) return 1;
        long n = _shape[0];
        for (int i = 1; i < _shape.length; i++) n *= _shape[i];
        return n;
    }

    /* ---------- End of public methods ---------- */


    /* Called by public constructors to initialize rank, shape, and stride */
    private void putShape(long[] shape) {
        _rank = shape.length;
        _shape = new long[_rank];
        _strides = new long[_rank];

        /* Using signed indices helps detect when index falls below 0. */
        for (int i = _rank - 1; i >= 0; i--) {
          _shape[i] = shape[i];
          if (i == _rank - 1)
            _strides[i] = 1;
          else
            _strides[i] = _strides[i+1] * _shape[i+1];
        }
    }

    /**
     * Constructor (For OMRunner only. Not intended for end user)
     *
     * @param data data buffer
     * @param shape data shape
     * @param dataType data type
     * @param endian data endian
     */
    protected OMTensor(ByteBuffer data, long[] shape, ByteOrder endian, int dataType) {
        if (dataType < 0 || dataType > LAST_ONNX_TYPE)
            throw new IllegalArgumentException(
                    "data type " + dataType + " unknown");
	if (endian == null)
	    throw new IllegalArgumentException("endian unknown");

        _data = data.order(endian);
        _dataType = dataType;
	putShape(shape);
    }

    /**
     * Constructor (For JNI wrapper only. Not intended for end user)
     *
     * @param data data buffer
     * @param shape data shape
     * @param strides data stride
     * @param dataType data type
     */
    protected OMTensor(ByteBuffer data, long[] shape, long[] strides, int dataType) {
        if (shape.length != strides.length)
            throw new IllegalArgumentException(
                    "shape.length (" + shape.length + ") != stride.length (" + strides.length + ")");
        if (dataType < 0 || dataType > LAST_ONNX_TYPE)
            throw new IllegalArgumentException(
                    "data type " + dataType + " unknown");
        /* data is owned by the native code. Make a copy to allow the JNI
           wrapper to clean up the native memory. */
        _data = ByteBuffer.allocateDirect(data.capacity()).order(nativeEndian);
        _data.slice().put(data.order(nativeEndian));
        _dataType = dataType;
        _rank = shape.length;
        _shape = shape;
        _strides = strides;
    }

    /**
     * Raw data getter (For JNI wrapper only. Not intended for end user)
     *
     * @return raw data
     */
    protected ByteBuffer getData() {
        return _data;
    }

    /**
     * Raw data setter (For JNI wrapper only. Not intended for end user)
     *
     * @param data raw data to be set
     */
    protected void setData(ByteBuffer data) {
        _data = data.order(nativeEndian);
    }
}
