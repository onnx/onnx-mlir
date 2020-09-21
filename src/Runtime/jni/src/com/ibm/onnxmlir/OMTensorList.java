package com.ibm.onnxmlir;

import java.util.HashMap;

public class OMTensorList {

    private OMTensor[] _rmrs;
    private HashMap<String, Integer> _n2i;
    
    /**
     * Constructor
     * 
     * @param rmrs DynMemRef array
     */
    public OMTensorList(OMTensor[] rmrs) {
        /* Go through the OMTensor array, check each for validity,
         * and create name (if not empty) to index mapping.
         */
        for (int i = 0; i < rmrs.length; i++) {
            if (rmrs[i] == null || !rmrs[i].isValidRmr())
                throw new IllegalArgumentException(
                        "OMTensor[" + i + "] is invalid");
            String name = rmrs[i].getName();
            if (!name.isEmpty() && _n2i.put(name, i) != null)
                throw new IllegalArgumentException(
                        "OMTensor[" + i + "] duplicate name: " + name);
        }

        _rmrs = rmrs;
    }

    /**
     * OMTensor array getter
     * 
     * @return OMTensor array
     */
    public OMTensor[] getRmrs() {
        return _rmrs;
    }
}
