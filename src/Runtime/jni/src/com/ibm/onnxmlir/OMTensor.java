package com.ibm.onnxmlir;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.ShortBuffer;

public class OMTensor {

    private final ByteOrder nativeEndian = ByteOrder.nativeOrder();
    
    /* We can use enum but that creates another class
     * which complicates things for JNI.
     */
    final int ONNX_TYPE_UNDEFINED  = 0;
    final int ONNX_TYPE_FLOAT      = 1;
    final int ONNX_TYPE_UINT8      = 2;
    final int ONNX_TYPE_INT8       = 3;
    final int ONNX_TYPE_UINT16     = 4;
    final int ONNX_TYPE_INT16      = 5;
    final int ONNX_TYPE_INT32      = 6;
    final int ONNX_TYPE_INT64      = 7;
    final int ONNX_TYPE_STRING     = 8;
    final int ONNX_TYPE_BOOL       = 9;
    final int ONNX_TYPE_FLOAT16    = 10;
    final int ONNX_TYPE_DOUBLE     = 11;
    final int ONNX_TYPE_UINT32     = 12;
    final int ONNX_TYPE_UINT64     = 13;
    final int ONNX_TYPE_COMPLEX64  = 14;
    final int ONNX_TYPE_COMPLEX128 = 15;
    final int ONNX_TYPE_BFLOAT16   = 16;

    final int[] ONNX_TYPE_SIZE = new int[] {
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
    };

    private ByteBuffer _data;
    private long[] _shape;
    private long[] _strides;
    private int _dataType;
    private int _rank;

    /**
     * Constructor
     */
    public OMTensor(byte[] data, long[] shape) {
        setByteData(data);
        putShape(shape);
    }

    public OMTensor(short[] data, long[] shape) {
        setShortData(data);
        putShape(shape);
    }

    public OMTensor(int[] data, long[] shape) {
        setIntData(data);
        putShape(shape);
    }

    public OMTensor(long[] data, long[] shape) {
        setLongData(data);
        putShape(shape);
    }

    public OMTensor(float[] data, long[] shape) {
        setFloatData(data);
        putShape(shape);
    }

    public OMTensor(double[] data, long[] shape) {
        setDoubleData(data);
        putShape(shape);
    }


    /* ---------- Byte data getter and setter ---------- */

    /**
     * Byte data getter
     *
     * @return byte data array
     */
    public byte[] getByteData() {
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
     * @param type data type to be set
     */
    public void setDataType(int dataType) {
        if (dataType < 0 || dataType > ONNX_TYPE_BFLOAT16)
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
     * Rank getter
     *
     * @return rank of the OMTensor
     */
    public int getRank() {
        return _rank;
    }

    /* ---------- Number of elements getter ---------- */

    /**
     * Number of elements getter
     * 
     * @return number of data elements in the data buffer
     */
    public long getNumElems() {
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

    /* For JNI wrapper only. Not intended for end user. */
    /**
     * Constructor
     * 
     * @param data data buffer
     * @param shape data shape
     * @param strides data stride
     * @param dataType data type
     */
    @SuppressWarnings("unused")
    private OMTensor(ByteBuffer data, long[] shape, long[] strides, int dataType) {
        if (shape.length != strides.length)
            throw new IllegalArgumentException(
                    "shape.length (" + shape.length + ") != stride.length (" + strides.length + ")");
        if (dataType < 0 || dataType > ONNX_TYPE_BFLOAT16)
            throw new IllegalArgumentException(
                    "data type " + dataType + " unknown");
        _data = data.order(nativeEndian);
        _dataType = dataType;
        _rank = shape.length;
        _shape = shape;
        _strides = strides;
    }

    /* For JNI wrapper only. Not intended for end user. */
    /**
     * Raw data getter
     * 
     * @return raw data
     */
    @SuppressWarnings("unused")
    private ByteBuffer getData() {
        return _data;
    }

    /* For JNI wrapper only. Not intended for end user. */
    /**
     * Raw data setter
     * 
     * @param data raw data to be set
     */
    @SuppressWarnings("unused")
    private void setData(ByteBuffer data) {
        _data = data.order(nativeEndian);
    }
}
