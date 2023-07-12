#!/usr/bin/env python3

import numpy as np
from PyCompileAndRuntime import OMCompileExecutionSession

# Load onnx model and create CompileExecutionSession object.
inputFileName = './mnist.onnx'
# Set the full name of compiled model
sharedLibPath = './mnist.so'
# Set the compile option as "-O3"
session = OMCompileExecutionSession(inputFileName,sharedLibPath,"-O3")
if session.get_compiled_result():
    print("error with :" + session.get_error_message())
    exit(1)
# Print the models input/output signature, for display.
# Signature functions for info only, commented out if they cause problems.
print("input signature in json", session.input_signature())
print("output signature in json",session.output_signature())

# Create an input arbitrarily filled of 1.0 values.
input = np.array([-0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.029637714847922325, 0.7467845678329468,
    1.7777715921401978, 2.796030282974243, 1.5104787349700928,
    1.5104787349700928, 0.36493754386901855, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, 0.09764464199542999, 2.5414655208587646,
    2.783302068710327, 2.796030282974243, 2.783302068710327,
    2.796030282974243, 2.783302068710327, 2.4141831398010254,
    0.6067740321159363, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, 0.4922199249267578, 2.5414655208587646,
    2.808758497238159, 1.7650433778762817, -0.4242129623889923,
    -0.4242129623889923, 2.0323362350463867, 2.796030282974243,
    2.808758497238159, 2.2869009971618652, -0.15692004561424255,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, 0.09764464199542999, 2.796030282974243,
    2.783302068710327, 1.2431857585906982, -0.2969306409358978,
    -0.4242129623889923, -0.4242129623889923, 0.4794916808605194,
    2.783302068710327, 2.796030282974243, 2.783302068710327,
    1.3831963539123535, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.2842023968696594, 1.7777715921401978,
    2.808758497238159, 2.5414655208587646, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    0.09764464199542999, 2.668747901916504, 2.808758497238159,
    2.796030282974243, 2.808758497238159, 1.7650433778762817,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, 0.22492696344852448,
    2.783302068710327, 2.796030282974243, 0.466763436794281,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, 2.159618616104126,
    1.4977505207061768, 2.5287373065948486, 2.796030282974243,
    2.783302068710327, 0.6195022463798523, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    1.5104787349700928, 2.796030282974243, 2.68147611618042,
    0.08491640537977219, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    0.09764464199542999, -0.4242129623889923, 0.6195022463798523,
    2.5541937351226807, 2.796030282974243, 2.5541937351226807,
    -0.042365945875644684, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    0.09764464199542999, 2.5414655208587646, 2.783302068710327,
    1.6377609968185425, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, 1.0013492107391357, 2.783302068710327,
    2.796030282974243, 1.7650433778762817, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, 0.8740668296813965, 2.808758497238159,
    2.796030282974243, 0.6195022463798523, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.15692004561424255,
    2.4141831398010254, 2.808758497238159, 2.796030282974243,
    0.36493754386901855, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, 1.3831963539123535,
    2.796030282974243, 2.783302068710327, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, 1.1286314725875854, 2.796030282974243,
    2.783302068710327, 0.8740668296813965, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.029637714847922325,
    2.5414655208587646, 2.808758497238159, 2.796030282974243,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    2.808758497238159, 2.796030282974243, 2.159618616104126,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    0.22492696344852448, 2.783302068710327, 2.796030282974243,
    2.783302068710327, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, 2.796030282974243, 2.783302068710327,
    2.159618616104126, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, 0.22492696344852448, 2.796030282974243,
    2.808758497238159, 2.796030282974243, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.15692004561424255, 2.808758497238159,
    2.796030282974243, 1.3831963539123535, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, 0.22492696344852448,
    2.783302068710327, 2.796030282974243, 2.783302068710327,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, 0.8740668296813965,
    2.796030282974243, 2.783302068710327, 0.8740668296813965,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, 1.905053973197937, 2.808758497238159,
    2.796030282974243, 1.6504892110824585, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.029637714847922325,
    2.5414655208587646, 2.808758497238159, 2.5414655208587646,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, 0.8740668296813965,
    2.796030282974243, 2.783302068710327, 2.668747901916504,
    0.08491640537977219, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    1.2559140920639038, 2.783302068710327, 2.796030282974243,
    0.466763436794281, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, 1.7777715921401978, 2.796030282974243,
    2.821486711502075, 2.796030282974243, 0.6195022463798523,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    0.6195022463798523, 2.808758497238159, 2.796030282974243,
    1.1413596868515015, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.2969306409358978,
    1.7650433778762817, 2.796030282974243, 2.783302068710327,
    2.668747901916504, 1.6377609968185425, 0.09764464199542999,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    1.6504892110824585, 2.668747901916504, 2.796030282974243,
    2.274172782897949, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, 1.1413596868515015,
    2.668747901916504, 2.808758497238159, 2.796030282974243,
    2.5541937351226807, 1.5104787349700928, 1.5232069492340088,
    2.5414655208587646, 2.808758497238159, 2.5414655208587646,
    1.1413596868515015, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, 0.08491640537977219, 1.4977505207061768,
    2.5287373065948486, 2.796030282974243, 2.783302068710327,
    2.796030282974243, 2.783302068710327, 1.4977505207061768,
    0.21219873428344727, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923, -0.4242129623889923,
    -0.4242129623889923, -0.4242129623889923], np.dtype(np.float32)).reshape(1,1,28,28)

# Run the model.
outputs = session.run(input)
# Analyze the output (first array in the list, of signature 1x10xf32).
prediction = outputs[0]
digit = -1
prob = 0.0
for i in range(0, 10):
    print("prediction ", i, "=", prediction[0, i])
    if prediction[0, i] > prob:
        digit = i
        prob = prediction[0, i]
# Print the value with the highest prediction (8 here).
print("The digit is", digit)