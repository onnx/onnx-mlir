package com.ibm.onnxmlir;

import java.util.HashMap;

public class OrderedRtMemRefDict {

    private RtMemRef[] _rmrs;
    private String[] _names;
    private HashMap<String, Integer> _n2i;
    
    /**
     * Constructor
     * 
     * @param rmrs DynMemRef array
     */
    public OrderedRtMemRefDict(RtMemRef[] rmrs) {
        this(rmrs, null);
    }

    /**
     * Constructor
     * 
     * @param rmrs DynMemRef array
     * @param names name array
     */
    public OrderedRtMemRefDict(RtMemRef[] rmrs, String[] names) {
        /* rmrs cannot be null or empty */
        if (rmrs == null || rmrs.length == 0)
            throw new IllegalArgumentException(
                    "Number of dmrs is invalid");
        
        /* If names is null or empty, construct a default one with
         * index as name.
         */
        if (names == null || names.length == 0) {
            names = new String[rmrs.length];
            for (int i = 0; i < rmrs.length; i++)
                names[i] = Integer.toString(i);
        }
        
        /* Number of rmrs and names must match */
        if (rmrs.length != names.length)
            throw new IllegalArgumentException(
                    "Number of dmrs and names do not match");

        /* Establish name to index mapping. Individual rmr is
         * checked for validity.
         */
        _n2i = new HashMap<String, Integer>();
        for (int i = 0; i < names.length; i++) {
            if (rmrs[i] == null || !rmrs[i].validRmr())
                throw new IllegalArgumentException(
                        "rmrs[" + i + "] is invalid");
            if (_n2i.put(names[i], i) != null)
                throw new IllegalArgumentException(
                        "name[" + i + "] = " + names[i] + " not unique");
        }
        _rmrs = rmrs;
        _names = names;
    }

    /**
     * RtMemRef getter by index
     * 
     * @param idx index of RtMemRef instance to get
     * @return RtMemRef instance
     */
    public RtMemRef getRmrbyIndex(int idx) {
        return _rmrs[idx];
    }

    /**
     * RtMemRef getter by name
     * 
     * @param name name of RtMemRef instance to get
     * @return RtMemRef instance
     */
    public RtMemRef getRmrByName(String name) {
        return _rmrs[_n2i.get(name)];
    }

    /**
     * RtMemRef array getter
     * 
     * @return RtMemRef array
     */
    public RtMemRef[] getRmrs() {
        return _rmrs;
    }

    /**
     * Name getter
     * 
     * @param idx index of name to get
     * @return name string
     */
    public String getName(int idx) {
        return _names[idx];
    }

    /**
     * Name array getter
     * 
     * @return name array
     */
    public String[] getNames() {
        return _names;
    }

    /**
     * RtMemRef array size getter
     * 
     * @return RtMemRef array size
     */
    public int size() {
        return _rmrs.length;
    }
}
