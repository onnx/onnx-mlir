package com.ibm.onnxmlir;

import java.util.HashMap;

public class RtMemRefList {

    private RtMemRef[] _rmrs;
    private HashMap<String, Integer> _n2i;
    
    /**
     * Constructor
     * 
     * @param rmrs DynMemRef array
     */
    public RtMemRefList(RtMemRef[] rmrs) {
        /* Go through the RtMemRef array, check each for validity,
         * and create name (if not empty) to index mapping.
         */
        for (int i = 0; i < rmrs.length; i++) {
            if (rmrs[i] == null || !rmrs[i].isValidRmr())
                throw new IllegalArgumentException(
                        "RtMemRef[" + i + "] is invalid");
            String name = rmrs[i].getName();
            if (!name.isEmpty() && _n2i.put(name, i) != null)
                throw new IllegalArgumentException(
                        "RtMemRef[" + i + "] duplicate name: " + name);
        }

        _rmrs = rmrs;
    }

    /**
     * RtMemRef array getter
     * 
     * @return RtMemRef array
     */
    public RtMemRef[] getRmrs() {
        return _rmrs;
    }
}
