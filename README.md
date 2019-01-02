# WebAI
API proposal for web based neural networks

```javascript

WebAI.getCapabilities()
  .then(capabilities => {
    /*
      {
        executionUnits: [
          {
            id: "CPU0",
            type: "cpu",
            dataTypes: ['fp16', 'fp32', 'fp64', 'u8', 'u16', 'u32', 'u64'] //data types that neural network uses to work
          },
          {
            id: "GPU0", //vulkan, openCL, GLSL
            type: "gpu",
            dataTypes: ['fp16', 'fp32', 'fp64', 'u8', 'u16', 'u32', 'u64'] //data types that neural network uses to work
          },
          {
            id: "DSP0",
            type: "dsp",
            dataTypes: ['fp16', 'fp32', 'fp64', 'u8', 'u16', 'u32', 'u64'] //data types that neural network uses to work
          },
          {
            id: "FPGA0",
            type: "fpga",
            dataTypes: ['fp16', 'fp32', 'fp64', 'u8', 'u16', 'u32', 'u64'] //data types that neural network uses to work
          }
        ]
      }
    */
  });



const ai = new WebAI.NeuralNetwork({
//example defaults:
  type: "NN", // RNN, LSTM, GRU
  timeStep: false,

  dataType: 'u8',
  activation: "tanh", // identity, binary, tanh, isrlu, relu, elu, softclip, sin, sinc, gauss - https://www.wikiwand.com/en/Activation_function
  layers: [8, 14, 8], // [ inputs, ... hidden ... , outputs ]

}, executionUnit);


const normalizeInput = input => { // normalize input data to expected data range: 0 .. 1
}

const normalizeOutput = output => { // normalize output data to expected data range: 0 .. 1
}


const data = ai.prepareData(normalizeInput, normalizeOutput, [
  /*
  inputData1, outputData1,
  inputData2, outputData2,
  ...
  inputDataN, outputDataN
  */
])

ai.train(data, options) // data can be a binary stream or typed array
  .then(trainingInfo => {
  });


ai.verify(data, options) // data can be a binary stream or typed array
  .then(verificationInfo => {
  });


ai.run(input, input => {}) //callback for input normalization, if skipped then input should be binary
  .then(output => {
  })

ai.toJson(json => {
}, options)

```