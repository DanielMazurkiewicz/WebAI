# WebAI
API proposal for web based neural networks.

## Basic goals

 - Providing simple to learn and use neural network API
 - Use of all possible and available hardware accelerations - all installed gpu's (vulkan, opencl, glsl, cuda), dsp's, fgpa's, attached AI devices, cpu's
 - Providing neural network models interoperatibility (including option of training NN on web based system and using model on low end 8 bit non web based systems)


## Javascript example

```javascript

WebAI.getCapabilities()
  .then(capabilities => {
    /*
      capabilities object should look like that:

      {
        executionUnits: [
          {
            id: "CPU0", //doesn't have to represent real CPU number, NN just have to be runned on separate cpu
            type: "cpu",
            dataTypes: ['fp16', 'fp32', 'fp64', 'u8', 'u16', 'u32', 'u64'] //data types that neural network uses to work
          },
          {
            id: "GPU0", //Vulkan, OpenCL, GLSL, CUDA
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
//when no object provided to constructor or missing properties, defaults should be assumed
  type: "NN", // RNN, LSTM, GRU
  timeStep: false,

  dataType: 'u8',
  activation: "tanh", // identity, binary, tanh, isrlu, relu, elu, softclip, sin, sinc, gauss - https://www.wikiwand.com/en/Activation_function
  layers: [8, 14, 8], // [ inputs, ... hidden ... , outputs ]
  setupData: '' // field optional, if string then base64 encoded setup data expected, can be also a array or typed array (of dataType)
}, executionUnit);


const normalizeInput = input => { // normalize input data to expected data range: 0 .. 1
  // normalization code
  return [ /* 0 , 1 , 0.5  .... */ ];
}

const normalizeOutput = output => { // normalize output data to expected data range: 0 .. 1
  //normalization code
  return [ /* 0 , 1 , 0.5  .... */ ];
}

const denormalizeOutput = outputNormalized => { // reverse output data normalization
  //denormalization code
}


//should prepare data according to neural network settings (especially number of inputs, number of outputs and data type)
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


ai.run(input, normalizeInput, denormalizeOutput) //if normalizeInput callback skipped then input should be typed array of dataType of NN, if denormalizeOutput skipped then typed array of dataType of NN should be returned
  .then(output => {
  })

ai.toJson(options)
  .then(json => {
    // json should contain all necesarry data to instantiate new WebAI.NeuralNetwork
  })

```

## Creadits

Heavily inspired on Brain.js