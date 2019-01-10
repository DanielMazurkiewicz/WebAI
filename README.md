# WebAI
API proposal for web based neural networks.

## Basic goals

 - Providing simple to learn and use machine learning API
 - Allowing to train and run AI in web browsers
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
            domains: ['core', 'http://w3c/tensorflow-2018.12', 'http://w3c/tensorflow-2019.06'],
            dataTypes: ['fp16', 'fp32', 'fp64', 'u8', 'u16', 'u32', 'u64'] //data types that neural network uses to work
            cue: 0  // number of machine learning tasks awaiting for execution
                    // there is no sense of executing in parallel NN tasks (run, train, valudate) on single execution unit
                    // every site should have own cue for every execution unit
                    // currently opened/visible site/tab should have a priority on executing NN tasks from cue over other non visible sites/tabs
          },
          {
            id: "GPU0", //Vulkan/SPIR-V, OpenCL, GLSL, CUDA
            type: "gpu",
            domains: ['core', 'http://w3c/tensorflow-2018.12', 'http://w3c/tensorflow-2019.06'],
            dataTypes: ['fp16', 'fp32', 'fp64', 'u8', 'u16', 'u32', 'u64'] //data types that neural network uses to work
            cue: 10
          },
          {
            id: "DSP0",
            type: "dsp",
            domains: ['core', 'http://w3c/tensorflow-2018.12', 'http://w3c/tensorflow-2019.06'],
            dataTypes: ['fp16', 'fp32', 'fp64', 'u8', 'u16', 'u32', 'u64'] //data types that neural network uses to work
            cue: 10
          },
          {
            id: "FPGA0",
            type: "fpga",
            domains: ['core', 'http://w3c/tensorflow-2018.12', 'http://w3c/tensorflow-2019.06'],
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

}, executionUnit);  // execution unit parameter could be provided as an array of execution units, 
                    // if so then JS engine should decide on what execution unit NN tasks will be executed 
                    // and it doesn't have to be same execution unit for all tasks


// === Start of user provided normalization/denormalization callbacks ===

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

//  === End of user provided normalization/denormalization callbacks ===




// Should prepare data according to neural network settings (especially number of inputs, number of outputs and data type)
// returns typed array of dataType of NN with series of instructions and data:
// <instruction> <inputData...> <outputData...>
//  where instruction could be encoded as:
//  0 - WebAI.reset translates to that code - no input and no output data, only performing reset
//  1 - WebAI.ignore translates to that code - expect only input data, no verification on output data
//  3 - full training data (input and output)
// Size of inputData and outputData determined by neural network architecture and data type
// Size of instruction is an equivalent of NN data type size

const data = ai.prepareData(normalizeInput, normalizeOutput, [
  /*
  inputData1, outputData1,
  inputData2, outputData2,
  ...
  WebAI.reset,                // const "reset" of WebAI defines an instruction inside data for training or verification procedures 
                              // that NN internal historical and recurrent data reset should be performed
  ...
  inputDataN, WebAI.ignore,   // const "ignore" of WebAI defines an instruction inside data for training or verification procedures 
                              // that for given input output should be ignored (for example in recurrent NNs)
  ...
  inputDataX, outputDataX
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

ai.reset() // resets any historical data inside NN (for NN's with recurences or historical data)

ai.moveTo(executionUnit) // moves ai (also if in ongoing operation - train, verify, run) to another execution unit


/*
  options example:
  {
    dataTypeAs: "base64",   // available options: base64, array, ?typedArray?
    layers: {               // if provided - exports only selected layers/selected range of layers as NN JSON object
                            // allows to make autoencoder NNs or to split NNs so they could be runned in chain of 
                            // separate sub-NNs across multiple execution units (eg. first 2 NN layers on CPU, 
                            // next 5 on GPU0 and another 4 on GPU1)
                            // should throw an error if there is any pipe that would make splitting NN impossible
      from: "input"
      to: "output"
    }
  }
  returns always fully functional NN model in JSON object
*/
ai.toJson(options)
  .then(json => {
    // json should contain all necesarry data to instantiate new WebAI.NeuralNetwork
  })


```

## Advanced neural networks architectures

 - For any sort of recurrent NN "pipes" could be used
 - Pipes could direct neuron outputs in both directions (forward and backward) - also to current layer/pipe as well
 - Activation function and other "future" options can be modified for every layer and every pipe in a layer
 - Pipes could also be assigned names
 - Layer could be build out of pipes only (as an option)
 - Pipes on first layer adds additional inputs
 - Pipes without property "to" pipe simply to next layer

 Properties summary:
  - **domain** property defines a domain for operations (??and activation functions??) in ml model, default: "core"
  - **name** property defines a name for layer or pipe
  - **type** property defines a type of operator for given layer or pipe, available options for core ml: neurons, lstm-n, lstm-p, gru. Where "n" stands for "normalized", and "p" for "pseudo". Extended operations sets might require defining aditional properties.
  - **activation** property defines an activation function for neurons in given layer or pipe
  - **pipes** property defines a list of pipes for given layer
  - **connections** property defines a way that nurons from given pipe/layer are connected to predecessing neurons
  - **count** property defines how many inputs/neurons/outputs is in layer/pipe
  - **to** property defines where to pipe neuron outputs(or data inputs if on input layer) of current layer/pipe
  - **history** property defines additional outputs of current layer/pipe build out of values of neuron outputs (or input data if on input layer) from given number of previous NN runs
  - **historyTo** property defines where to pipe previous NN runs values of neuron outputs(or data inputs if on input layer) from current layer/pipe

```javascript
const ai = new WebAI.NeuralNetwork({
  dataType: 'u8',
  activation: "tanh", // if more sophisticated architecture provided this is considered only as default activation

  layers: [
    {
      name: "input", //optional - allows later to pipe by layer name
      count: 8
    },
    8, // regular layer with 8 neurons
    10,
    {
      count: 4,   // 4 outputs
      pipes: [    // multiple pipes posible
        {
          activation: "isrlu",    // overwrites default activation function
          count: 16,              // adds 16 neurons to current layer of which outputs will be available at piped layers/pipes
          to: ['input', 1]        // piped to named layer (or pipe) and by index of layer
        }
      ]
    }
  ],

}, executionUnit);

```

### Another advanced example with usage of "history" shown


```javascript
const ai = new WebAI.NeuralNetwork({
  dataType: 'fp32',
  activation: "tanh",

  layers: [
    { // first layer always is an input layer
      name: 'input'
      pipes: [
        {
          name: 'rgb',
          count: 3,
          history: 2,           // reserves memory for 6 additional values (count * history = 2 * 3 = 6) 
                                // to keep two previous values of given 3 inputs
          historyTo: ['input']  // pipe historical values to input layer (will create hidden 6 inputs)
                                // note that original 3 inputs are still piped directly to next layer
        },
        {
          name: 'coordinates',
          count: 2
        }
      ]
    },
    {
      // layer of GRU LSTM kind of neurons
      connections: "all-to-all", // this is default and can be skipped, 
                                 // every neuron in this layer has a connection to every predecessing neuron 
                                 // output (or every input value if predecessing is input layer/input pipe)

      type: "gru",  // default value if property skipped is: "neurons"
                    // other available options: "lstm-p", "lstm-n", "gru" (core operators)

      count: 3      // 3 neurons of GRU LSTM
    },
    18, // regular layer with 18 neurons
    3   // 3 neurons output layer, last layer always is an output layer
  ],

}, executionUnit);

```

## API for low level operations

```javascript

WebAI.getOperations("operations_domain eg: 'core'", executionUnit)
  .then(webml => {
    /*
      would return object with operation functions:
      {
        ...
        fusedMatMul(a, b, transposeA, transposeB, bias?, activationFn?) {}
        ...
      }
    */

    webml.fusedMatMul(a, b, transposeA, transposeB, bias?, activationFn?);

  })


// custom domains will be available only for cpu execution units.
WebAI.defineCustomDomain("domain_name");


// defines a new operation for given domain, operation execution always fallbacks to JS engine, 
// no matter of what execution unit is used
WebAI.defineCustomOperation(
  "operation_name", 
  "operation_domain", 
  function operation(operation_params) {
  // js code itself can make use of any existing API (including webGPU and webgl)

  },
  ["parameter1", "parameter2", "parameterN"], // JSON model properties that should be passed as arguments to this operation
  // there should be a list of names reserved for special purpouses
);

// defines a new activation function for given domain, its execution always fallbacks to JS engine
WebAI.defineCustomActivation("activation_name", "activation_domain", function activation(activationInput) {
  // ...
  return activationOutput;
});


```

## Further things to keep in mind

There could be also designed standardized JSON format for simple normalizations/denormalizations of numbers and enumerations

## Links
https://towardsdatascience.com/a-deeper-understanding-of-nnets-part-1-cnns-263a6e3ac61

https://becominghuman.ai/a-deeper-understanding-of-nnets-part-2-rnns-b32240998fa9

https://medium.com/@godricglow/a-deeper-understanding-of-nnets-part-3-lstm-and-gru-e557468acb04



https://www.wikiwand.com/en/Activation_function

https://webmachinelearning.github.io/

https://webmachinelearning.github.io/webnn/



## Credits

Heavily inspired on Brain.js