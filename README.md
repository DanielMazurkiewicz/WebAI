# WebAI
API proposal for web based neural networks.

## Basic goals

 - Providing simple to learn and use machine learning API
 - Allowing to use of all possible and available hardware accelerations (simultaneusly) for training and running NNs - all installed gpu's (vulkan, opencl, glsl, cuda), dsp's, fgpa's, attached AI devices, cpu's
 - Neural networks architecture as flexible descriptive JSON format that can be:
    * easily maintained by developers
    * easily compiled into executable code 
    * easily transferred between proceeses or machines
    * easily converted into a code in another programming languague
    * flexible enough so existing models from tensorflow or caffe or other popular libraries could be easily converted
    * easily used by visual neural network designing/training tools


## Javascript example

This is API proposal - just something to start with standardization process.

```javascript

WebAI.getCapabilities(optionalCapabilitiesQueryObject)
  .then(capabilities => {
    /*
      capabilities object could look like that:

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



const ai = new WebAI.NeuralNetwork({ // simple NN example object
  // when no object provided to constructor or missing properties, defaults should be assumed

  dataType: 'fp16',
  activation: "tanh", // available options: identity, binary, tanh, isrlu, relu, elu, softclip, sin, sinc, gauss
                      // see https://www.wikiwand.com/en/Activation_function

  layers: [8, 14, 8], // [ inputs, ... hidden ... , outputs ]

  setupData: '' // weights, biases ... - field optional,
                // if string then base64 encoded setup data expected, 
                // can be also an array or typed array (of dataType),
                // if field not provided random values assumed

}, executionUnit);




const normalizeInput = input => { // normalize input data to expected data range: 0 .. 1
  // normalization code
  return [ /* 0 , 1 , 0.5  .... */ ];
}

const normalizeOutput = output => { // normalize output data to expected data range: 0 .. 1
  // normalization code
  return [ /* 0 , 1 , 0.5  .... */ ];
}

const denormalizeOutput = outputNormalized => { // reverse output data normalization
  // outputNormalized is an array of floats with range 0.0 to 1.0
  // denormalization code
  // returns denormalized output
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
/* optionally:
  const stream = ai.prepareDataStream(normalizeInput, normalizeOutput)
*/


ai.train(data, options) // data can be a binary stream or typed array
  /*
    example options object could look like that:
    {
      type: "back-propagation"
      ... -  and some backpropagation training detailed options
    }
  */
  .then(trainingInfo => {
  });


ai.stopTraining() // stops training (if ongoing) and fires promise for ai.train


ai.verify(data, options) // data can be a binary stream or typed array
  .then(verificationInfo => {
  });


ai.run(input, normalizeInput, denormalizeOutput) 
  // if normalizeInput callback not provided or null then input expected to be a typed array of dataType of NN, 
  // if denormalizeOutput not provided or null then typed array of dataType of NN should be returned

  .then(output => {
  })

ai.moveTo(executionUnit) // moves ai (also if in ongoing operation - train, verify, run) to another execution unit



/*
  options example:
  {
    dataTypeAs: "base64" // available options: base64, array
  }
*/
ai.toJson(options)
  .then(json => {
    // json should contain all necesarry data to instantiate new WebAI.NeuralNetwork
  })


```

## Advanced neural networks architectures

 * For any sort of recurrent NN "pipes" could be used
 * Pipes could direct neuron outputs in both directions (forward and backward) and to current layer as well
 * Activation function and other "future" options can be modified for every layer and every pipe in a layer


```javascript
const ai = new WebAI.NeuralNetwork({
  dataType: 'u8',
  activation: "tanh", // if more sophisticated architecture provided this is considered only as default activation

  layers: [
    {
      name: "input", //optional - allows to pipe by name
      count: 8
    },
    8, // regular layer with 8 neurons
    10,
    {
      count: 4, // 4 outputs
      pipe: [   // multiple pipes posible
        {
          activation: "isrlu",    // overwrites default activation function
          count: 16,              // adds 16 neurons to current layer of which outputs will be available at piped layers
          toLayers: ['input', 1]  // piped to named layer and by index of layer
        }
      ]
    }
  ],

}, executionUnit);

```

## Links
https://towardsdatascience.com/a-deeper-understanding-of-nnets-part-1-cnns-263a6e3ac61

https://www.wikiwand.com/en/Activation_function

https://webmachinelearning.github.io/

https://webmachinelearning.github.io/webnn/



## Credits

Heavily inspired on Brain.js