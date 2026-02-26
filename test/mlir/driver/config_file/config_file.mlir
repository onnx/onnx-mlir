// RUN: out=$(dirname %s)/test1 && onnx-mlir -v %s -o $out| FileCheck --check-prefix=DEFAULT_CONFIG_FILE %s && rm ${out}.so
// RUN: out=$(dirname %s)/test2 && onnx-mlir -v --config-file=$(dirname %s)/custom-omconfig.json %s -o $out| FileCheck --check-prefix=CUSTOM_CONFIG_FILE %s && rm ${out}.so
// RUN: out=$(dirname %s)/test3 && onnx-mlir -v -O3 %s -o $out| FileCheck --check-prefix=OVERWRITE_CONFIG_FILE %s && rm ${out}.so
// RUN: out=$(dirname %s)/test4 && onnx-mlir -v --config-file $(dirname %s)/custom-omconfig.json %s -o $out| FileCheck --check-prefix=PARSE_CONFIG_FILE_1 %s && rm ${out}.so
// RUN: out=$(dirname %s)/test5 && onnx-mlir -v -config-file $(dirname %s)/custom-omconfig.json %s -o $out| FileCheck --check-prefix=PARSE_CONFIG_FILE_2 %s && rm ${out}.so

module {
  func.func @main_graph(%arg0: tensor<?x?x?xf32>) -> (tensor<?x?x?xf32>) {
    %0 = "onnx.Relu"(%arg0) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    onnx.Return %0 : tensor<?x?x?xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

// COM: Check if the driver reads the default omconfig.json in the same folder of the input model.
// DEFAULT_CONFIG_FILE-DAG: Config file: {{.*}}omconfig.json 
// DEFAULT_CONFIG_FILE-DAG: Onnx-mlir command: {{.*}}onnx-mlir -v {{.*}}.mlir -o {{.*}}test1 -O2 --shapeInformation=0:3x4x5
// DEFAULT_CONFIG_FILE-DAG: {{.*}}opt -O2{{.*}}
// DEFAULT_CONFIG_FILE-DAG: {{.*}}llc -O2{{.*}}

// COM: Check the compile options --config-file 
// CUSTOM_CONFIG_FILE-DAG: Config file: {{.*}}custom-omconfig.json 
// CUSTOM_CONFIG_FILE-DAG: Onnx-mlir command: {{.*}}onnx-mlir -v --config-file={{.*}}custom-omconfig.json {{.*}}.mlir -o {{.*}}test2 -O3 --shapeInformation=0:3x4x5
// CUSTOM_CONFIG_FILE-DAG: {{.*}}opt -O3{{.*}}
// CUSTOM_CONFIG_FILE-DAG: {{.*}}llc -O3{{.*}}

// COM: Check if the compile options from the config file overwrites the commandline options.
// OVERWRITE_CONFIG_FILE-DAG: Config file: {{.*}}omconfig.json 
// OVERWRITE_CONFIG_FILE-DAG: Onnx-mlir command: {{.*}}onnx-mlir -v {{.*}}.mlir -o {{.*}}test3 -O2 --shapeInformation=0:3x4x5
// OVERWRITE_CONFIG_FILE-DAG: {{.*}}opt -O2{{.*}}
// OVERWRITE_CONFIG_FILE-DAG: {{.*}}llc -O2{{.*}}

// COM: Check parsing the compile options "--config-file config.json" 
// PARSE_CONFIG_FILE_1-DAG: Config file: {{.*}}custom-omconfig.json 
// PARSE_CONFIG_FILE_1-DAG: Onnx-mlir command: {{.*}}onnx-mlir -v --config-file {{.*}}custom-omconfig.json {{.*}}.mlir -o {{.*}}test4 -O3 --shapeInformation=0:3x4x5
// PARSE_CONFIG_FILE_1-DAG: {{.*}}opt -O3{{.*}}
// PARSE_CONFIG_FILE_1-DAG: {{.*}}llc -O3{{.*}}

// COM: Check parsing the compile options "-config-file config.json" 
// PARSE_CONFIG_FILE_2-DAG: Config file: {{.*}}custom-omconfig.json 
// PARSE_CONFIG_FILE_2-DAG: Onnx-mlir command: {{.*}}onnx-mlir -v -config-file {{.*}}custom-omconfig.json {{.*}}.mlir -o {{.*}}test5 -O3 --shapeInformation=0:3x4x5
// PARSE_CONFIG_FILE_2-DAG: {{.*}}opt -O3{{.*}}
// PARSE_CONFIG_FILE_2-DAG: {{.*}}llc -O3{{.*}}
