package com.ibm.onnxmlir;

import java.util.HashMap;

public class OMTensorList {

    private OMTensor[] _omts;
    private HashMap<String, Integer> _n2i;
    
    /**
     * Constructor
     * 
     * @param omts DynMemRef array
     */
    public OMTensorList(OMTensor[] omts) {
        /* Go through the OMTensor array, check each for validity,
         * and create name (if not empty) to index mapping.
         */
        for (int i = 0; i < omts.length; i++) {
            if (omts[i] == null || !omts[i].isValidOmt())
                throw new IllegalArgumentException(
                        "OMTensor[" + i + "] is invalid");
            String name = omts[i].getName();
            if (!name.isEmpty() && _n2i.put(name, i) != null)
                throw new IllegalArgumentException(
                        "OMTensor[" + i + "] duplicate name: " + name);
        }

        _omts = omts;
    }

    /**
     * OMTensor array getter
     * 
     * @return OMTensor array
     */
    public OMTensor[] getOmts() {
        return _omts;
    }
}
