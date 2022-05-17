// SPDX-License-Identifier: Apache-2.0

package com.ibm.onnxmlir;

/**
 * Class holding an array of OMTensor so that they can
 * be passed into and out of the compiled model runtime
 * entry point as inputs and outputs.
 */
public class OMTensorList {

    private OMTensor[] _omts;
    
    /**
     * Constructor
     * 
     * @param omts OMTensor array
     */
    public OMTensorList(OMTensor[] omts) {
        _omts = omts;
    }

    /**
     * OMTensor array getter
     * 
     * @return OMTensor array
     */
    public OMTensor[] getOmtArray() {
        return _omts;
    }
    
    /**
     * OMTensor getter
     * 
     * @param index index of OMTensor to get
     * @return OMTensor at index specified
     */
    public OMTensor getOmtByIndex(int index) {
        return _omts[index];
    }
}
