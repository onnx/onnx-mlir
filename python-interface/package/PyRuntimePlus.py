#!/usr/bin/env python3
#python functions for PyRuntimePlus

import numpy as np
from PyRuntime import ExecutionSession
from PyOnnxMlirCompiler import OnnxMlirCompiler, OnnxMlirTarget, OnnxMlirOption

class OMCompilerOption:

    def set_target_triple(self,target_triple):
        self.target_triple = target_triple
    
    def get_target_triple(self):
        return(self.target_triple)
    
    def set_target_arch(self,target_arch):
        self.target_arch = target_arch
    
    def get_target_arch(self):
        return(self.target_arch)
    
    def set_target_cpu(self,target_cpu):
        self.target_cpu = target_cpu
    
    def get_target_cpu(self):
        return(self.target_cpu)
    
    def set_target_accel(self,target_accel):
        self.target_accel = target_accel
    
    def get_target_accel(self):
        return(self.target_accel)
    
    def set_opt_level(self,level):
        self.opt_level = level
    
    def get_opt_level(self,level):
        return(self.opt_level)

    def set_opt_flag(self,opt_flag):
        self.opt_flag = opt_flag
    
    def get_opt_flag(self):
        return(self.opt_flag)

    def set_llc_flag(self,llc_flag):
        self.llc_flag = llc_flag
    
    def get_llc_flag(self):
        return(self.llc_flag)

    def set_llvm_flag(self,llvm_flag):
        self.llvm_flag = llvm_flag
    
    def get_llvm_flag(self):
        return(self.llvm_flag)
    
    def set_verbose(self,verbose):
        self.verbose = verbose
    
    def get_verbose(self):
        return(self.verbose)

    def set_target(self,target):
        self.target = target
    
    def get_target(self,target):
        return(self.target)

class OMSession:

    def _init_(self,file):
        self.file = file
        self.compiler = PyOnnxMlirCompiler.OnnxMlirCompiler(file)
    
    def compile(self,option):
        if(option.get_target_triple()!=""):
            self.compiler.set_option(PyOnnxMlirCompiler.OnnxMlirOption.target_triple, option.get_target_triple())
        
        if(option.get_target_arch()!=""):
            self.compiler.set_option(PyOnnxMlirCompiler.OnnxMlirOption.target_arch, option.get_target_arch())

        if(option.get_target_cpu()!=""):
            self.compiler.set_option(PyOnnxMlirCompiler.OnnxMlirOption.target_cpu, option.get_target_cpu())

        if(option.get_target_accel()!=""):
            self.compiler.set_option(PyOnnxMlirCompiler.OnnxMlirOption.target_accel, option.get_target_accel())

        if(option.get_opt_level()!=""):
            self.compiler.set_option(PyOnnxMlirCompiler.OnnxMlirOption.opt_level, option.get_opt_level())

        if(option.get_opt_flag()!=""):
            self.compiler.set_option(PyOnnxMlirCompiler.OnnxMlirOption.opt_flag, option.get_opt_flag())

        if(option.get_llc_flag()!=""):
            self.compiler.set_option(PyOnnxMlirCompiler.OnnxMlirOption.llc_flag, option.get_llc_flag())

        if(option.get_verbose()!=""):
            self.compiler.set_option(PyOnnxMlirCompiler.OnnxMlirOption.verbose, option.get_verbose())
        
        rc = self.compiler.compile(option.get_target(), PyOnnxMlirCompiler.OnnxMlirTarget.emit_lib)
        self.output_file_name = self.compiler.get_output_file_name()
        return rc
    
    def run(self,input):
        self.session = PyRuntime.ExecutionSession(self.output_file_name)
        outputs = self.session.run([input])
        return outputs
    
    def print_input_signature(self):
        print("input signature in json", self.session.input_signature())
    
    def print_output_signature(self):
        print("output signature in json",self.session.output_signature())