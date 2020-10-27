package com.ibm.onnxmlir;

import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.ShortBuffer;

public class OMTensor {
    final ByteOrder endian = ByteOrder.nativeOrder();
    
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

    private ByteBuffer _allocatedPtr;
    private long[] _shape;
    private long[] _stride;
    private int _dataType;
    private int _rank;
    private String _name;
    
    /**
     * Constructor
     */
    public OMTensor(int rank) {
        if (rank <= 0)
            throw new IllegalArgumentException(
                    "invalid rank " + rank);
        _allocatedPtr = null;
        _shape = new long[rank];
        _stride = new long[rank];
        _dataType = ONNX_TYPE_UNDEFINED;
        _rank = rank;
        _name = "";
    }

    /* ---------- Raw data getter and setter ---------- */
    /* For JNI wrapper only. Not intended for end user. */

    /**
     * Raw data getter
     * 
     * @return raw data
     */
    @SuppressWarnings("unused")
    private ByteBuffer getData() {
        return _allocatedPtr;
    }

    /**
     * Raw data setter
     * 
     * @param data raw data to be set
     */
    @SuppressWarnings("unused")
    private void setData(ByteBuffer data) {
        _allocatedPtr = data.order(endian);
    }
    
    /* ---------- Byte data getter and setter ---------- */

    /**
     * Byte data getter
     *
     * @return byte data array
     */
    public byte[] getByteData() {
        if (_allocatedPtr == null) return null;

        /* asReadOnlyBuffer() creates a new view so the position of the
         * original data will stay at 0 for subsequent getByteData()
         * after get(b).
         */
        byte[] b = new byte[_allocatedPtr.limit()];
        _allocatedPtr.asReadOnlyBuffer().get(b);
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
        _allocatedPtr = ByteBuffer.allocateDirect(data.length);
        _allocatedPtr.slice().put(data);
        _dataType = ONNX_TYPE_INT8;
    }

    /* ---------- Short data getter and setter ---------- */

    /**
     * Short data getter
     * 
     * @return short data array
     */
    public short[] getShortData() {
        if (_allocatedPtr == null) return null;

        /* asShortBuffer() creates a new view so the position of the
         * original data will stay at 0 for subsequent getShortData()
         * after get(s).
         */
        ShortBuffer sb = _allocatedPtr.asShortBuffer();
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
        _allocatedPtr = ByteBuffer.allocateDirect(data.length*2).order(endian);
        _allocatedPtr.asShortBuffer().put(data);
        _dataType = ONNX_TYPE_INT16;
    }

    /* ---------- Int data getter and setter ---------- */

    /**
     * Int data getter
     * 
     * @return int data array
     */
    public int[] getIntData() {
        if (_allocatedPtr == null) return null;

        /* asIntBuffer() creates a new view so the position of the
         * original data will stay at 0 for subsequent getIntData()
         * after get(i).
         */
        IntBuffer ib = _allocatedPtr.asIntBuffer();
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
        _allocatedPtr = ByteBuffer.allocateDirect(data.length*4).order(endian);
        _allocatedPtr.asIntBuffer().put(data);
        _dataType = ONNX_TYPE_INT32;
    }

    /* ---------- Long data getter and setter ---------- */

    /**
     * Long data getter
     * 
     * @return long data array
     */
    public long[] getLongData() {
        if (_allocatedPtr == null) return null;

        /* asLongBuffer() creates a new view so the position of the
         * original data will stay at 0 for subsequent getLongData()
         * after get(l).
         */
        LongBuffer lb = _allocatedPtr.asLongBuffer();
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
        _allocatedPtr = ByteBuffer.allocateDirect(data.length*8).order(endian);
        _allocatedPtr.asLongBuffer().put(data);
        _dataType = ONNX_TYPE_INT64;
    }

    /* ---------- Float data getter and setter ---------- */

    /**
     * Float data getter
     * 
     * @return float data array
     */
    public float[] getFloatData() {
        if (_allocatedPtr == null) return null;

        /* asFloatBuffer() creates a new view so the position of the
         * original data will stay at 0 for subsequent getFloatData()
         * after get(f).
         */
        FloatBuffer fb = _allocatedPtr.asFloatBuffer();
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
        _allocatedPtr = ByteBuffer.allocateDirect(data.length*4).order(endian);
        _allocatedPtr.asFloatBuffer().put(data);
        _dataType = ONNX_TYPE_FLOAT;
    }

    /* ---------- Double data getter and setter ---------- */

    /**
     * Double data getter
     * 
     * @return double data array
     */
    public double[] getDoubleData() {
        if (_allocatedPtr == null) return null;

        /* asDoubleBuffer() creates a new view so the position of the
         * original data will stay at 0 for subsequent getDoubleData()
         * after get(d).
         */
        DoubleBuffer db = _allocatedPtr.asDoubleBuffer();
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
        _allocatedPtr = ByteBuffer.allocateDirect(data.length*8).order(endian);
        _allocatedPtr.asDoubleBuffer().put(data);
        _dataType = ONNX_TYPE_DOUBLE;
    }

    /* ---------- Data sizes getter and setter ---------- */

    /**
     * Data sizes getter
     *
     * @return data sizes array
     */
    public long[] getDataSizes() {
        return _shape;
    }

    /**
     * Data sizes setter
     *
     * @param dataSizes data sizes array to be set
     */
    public void setDataSizes(long[] dataSizes) {
        if (dataSizes.length != _rank)
            throw new IllegalArgumentException(
                    "array length " + dataSizes.length + " != rank " + _rank);
        _shape = dataSizes.clone();
    }

    /* ---------- Data strides getter and setter ---------- */

    /**
     * Data strides getter
     * 
     * @return data strides array
     */
    public long[] getDataStrides() {
        return _stride;
    }

    /**
     * Data strides setter
     * 
     * @param dataStrides data strides array to be set
     */
    public void setDataStrides(long[] dataStrides) {
        if (dataStrides.length != _rank)
            throw new IllegalArgumentException(
                    "array length " + dataStrides.length + " != rank " + _rank);
        _stride = dataStrides.clone();
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
    public long getDataBufferSize() {
        return _allocatedPtr == null ? 0 : _allocatedPtr.limit();
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

    /* ---------- Name getter and setter ---------- */

    /**
     * Name getter
     *
     * @return name of the OMTensor
     */
    public String getName() {
        return _name;
    }

    /**
     * Name setter
     *
     * @param name name of the OMTensor
     */
    public void setName(String name) {
        _name = name == null ? "" : name;
    }

    /**
     * Number of elements getter
     * 
     * @return number of data elements in the data buffer
     */
    public long getNumOfElems() {
        long n = _shape[0];
        for (int i = 1; i < _shape.length; i++) n *= _shape[i];
        return n;
    }

    /**
     * Check validity of OMTensor
     * 
     * @return true if OMTensor is valid, false otherwise
     */
    public boolean isValidOmt() {
        return (_allocatedPtr != null &&
                _allocatedPtr.limit() != 0 &&
                _allocatedPtr.limit() == getNumOfElems() * ONNX_TYPE_SIZE[_dataType]);
    }
}
