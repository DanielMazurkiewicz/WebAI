# WebAI
API proposal for web based neural networks.

## Basic goals

 - Providing simple to learn and use neural network API
 - Allowing to use of all possible and available hardware accelerations - all installed gpu's (vulkan, opencl, glsl, cuda), dsp's, fgpa's, attached AI devices, cpu's
 - Providing neural network models interoperatibility (including option of training NN on web based system and using model on low end 8 bit non web based systems)


## Javascript example

```javascript

WebAI.getCapabilities(optionalCapabilitiesQueryObject)
  .then(capabilities => {
    /*
      capabilities object should look like that:

      {
        executionUnits: [
          {
            id: "CPU0", //doesn't have to represent real CPU number, NN just have to be runned on separate cpu
            type: "cpu",
            dataTypes: ['fp16', 'fp32', 'fp64', 'u8', 'u16', 'u32', 'u64'] //data types that neural network uses to work
            cue: 0 //number of machine learning tasks awaiting for execution
          },
          {
            id: "GPU0", //Vulkan, OpenCL, GLSL, CUDA
            type: "gpu",
            dataTypes: ['fp16', 'fp32', 'fp64', 'u8', 'u16', 'u32', 'u64'] //data types that neural network uses to work
            cue: 10
          },
          {
            id: "DSP0",
            type: "dsp",
            dataTypes: ['fp16', 'fp32', 'fp64', 'u8', 'u16', 'u32', 'u64'] //data types that neural network uses to work
            cue: 10
          },
          {
            id: "FPGA0",
            type: "fpga",
            dataTypes: ['fp16', 'fp32', 'fp64', 'u8', 'u16', 'u32', 'u64'] //data types that neural network uses to work
            cue: 10
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
  setupData: '' // weights, biases ... - field optional, if string then base64 encoded setup data expected, can be also a array or typed array (of dataType)
}, executionUnit);



//advanced NN architecture:
const ai = new WebAI.NeuralNetwork({
  type: "NN", // RNN, LSTM, GRU
  timeStep: false,

  dataType: 'u8',
  activation: "tanh", // identity, binary, tanh, isrlu, relu, elu, softclip, sin, sinc, gauss - https://www.wikiwand.com/en/Activation_function

  layers: [
    {
      name: "input", //optional - allows to pipe by name
      count: 8
    },
    8, // regular layer with 8 neurons
    10,
    {
      count: 4, // 4 outputs
      pipe: [ // multiple pipes posible
        {
          count: 16, // adds 16 neurons to current layer of which outputs will be available at piped layers
          toLayers: ['input', 1] // piped to named layer and by index of layer
        }
      ]
    }
  ],

  setupData: '' // weights, biases ... - field optional, if string then base64 encoded setup data expected, can be also a array or typed array (of dataType)
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


ai.stopTraining() // stops training (if ongoing) and fires promise for ai.train


ai.verify(data, options) // data can be a binary stream or typed array
  .then(verificationInfo => {
  });


ai.run(input, normalizeInput, denormalizeOutput) //if normalizeInput callback skipped then input should be typed array of dataType of NN, if denormalizeOutput skipped then typed array of dataType of NN should be returned
  .then(output => {
  })

ai.moveTo(executionUnit) // moves ai (also if in ongoing operation) to another execution unit

ai.toJson(options)
  .then(json => {
    // json should contain all necesarry data to instantiate new WebAI.NeuralNetwork
  })


```

## Credits

Heavily inspired on Brain.js