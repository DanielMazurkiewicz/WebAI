# WebAI
API proposal for web based neural networks. It is still a form of notes and drafts, but already pretty mature.
Follow the status of machine learning standardization process:
 * https://webmachinelearning.github.io/
 * https://github.com/webmachinelearning


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
    * suitable for basic and for scientific purpouses

## TLDR;

 * WebAI.getCapabilities()
 * WebAI.getOperations(operationsQuery)

 * WebAI.defineDomain(domainDescriptionObject);
 * WebAI.defineOperation(operationDescriptionObject, operation)
 * WebAI.defineOperationVariant(operationVariantDescriptionObject, operation)
 * WebAI.defineExport(...TBD...)
 * WebAI.defineRun(...TBD...)
 * WebAI.defineTrain(...TBD...)
 * WebAI.defineVerify(...TBD...)

 * const ai = new WebAI.NeuralNetwork(nnModelObject)
 * ai.prepareData(normalizeInput, normalizeOutput, arrayOfData)
 * ai.train(data, options)
 * ai.stopTraining()
 * ai.verify(data, options)
 * ai.run(input, normalizeInput, denormalizeOutput, options)
 * ai.reset()
 * ai.getInternalState()
 * ai.setInternalState(state)
 * ai.export(options)


## Simple self explanatory JavaScript example


```javascript

WebAI.getCapabilities(optionalCapabilitiesQueryObject)
  .then(capabilities => {
    /*
      capabilities object could look like that:
      {
          domains: {
            'core': ['fp16', 'fp32', 'fp64', 'u8', 'u16', 'u32', 'u64'], // data types that neural network can use to work
                                                                         // determined by hardware and software implementations
                                                                         // of all hardware devices available to browser
            'http://w3c/tensorflow-2018.12': ['fp16', 'fp32', 'fp64', 'u8', 'u16', 'u32', 'u64'],
            'http://w3c/onnx-2019.06': ['fp16', 'fp32', 'fp64', 'u8', 'u16', 'u32', 'u64'],
          }
      }
    */
  });



const ai = new WebAI.NeuralNetwork({ // simple NN example object
  // when no object provided to constructor or missing properties, defaults should be assumed
  domain: 'core',     // can be skipped because it is default domain, if given domain is not supported then 
                      // throws error

  dataType: 'fp16',   // every domain can define its own default data type, and if it doesn't, lack of this
                      // property throws error

  minMops: 300,       // optional field, if provided informs Javascript about expected "millions of operations per second"
                      //  on given data type (helps to determine hardware device to run NN)
  activation: "tanh", // Sets default activation function, unless staten another, every layer and pipe will be using
                      // given activation function
                      // If parameter ommited, then activation will be used only if defined at layers or pipes

                      // To consider activations: identity, binary, tanh, isrlu, relu, elu, softclip, sin, sinc, gauss,
                      // see https://www.wikiwand.com/en/Activation_function

  layers: [8, 14, 8], // [ inputs, ... hidden ... , outputs ]

  setup: {    // field optional

    data: '', // Weights, biases, and other parameters ...
              // If string then base64 encoded setup data expected, 
              // Can be also an array or typed array (of dataType),
              // If field not provided random values assumed

  }
});   


// === Start of user provided normalization/denormalization callbacks ===

const normalizeInput = input => { // normalize input data to expected data range: 0 .. 1
  // normalization code
  return [ /* 0 , 1 , 0.5  .... */ ];
}
/*
  function can return:
    * array of numbers, 
    * typed array,
    * object - if provided, then its properties values should be arrays that will be
        passed directly to named pipes on input layer where names will correspond to
        named properties, all values that will be not in object will be passed to
        unnamed pipes and input layer in order of appearance
    * array of mixed above - will be automatically unwrapped to single array
  
  Examples:
    * return { imageData: [...], coordinates: [0, 0]};  // can return just an object
    * return [{ imageData: [...], coordinates: [0, 0]}, [0, 0, [1]], 0, 0, 0.6]; // or mixed data
*/


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



// training can be performed only on native operations and activations of "core" domain
// otherwise throws error
ai.train(data, options) // data can be a binary stream or typed array
  /*
    example options object could look like that:
    {
      toSkip: ["names_of_pipes_or_layers"],   // optional and if provided performs training without changing data from given
                                              // layers or pipes

      toTrain: ["names_of_pipes_or_layers"],  // optional and if provided performs training only on given layers or pipes

      // training stop conditions:

      maxIterations: 20000,                   // the maximum times to iterate the training data (default: Infinity)
      timeout: 1000,                          // the max number of milliseconds to train for (default: Infinity)
      errorMeasureType: "mean-square",        // default: "mean-square"
      errorThreshold: 0.5,                    // the acceptable error percentage from training data --> number between 0 and 100
                                              // (default: 0)
                                              //
                                              // If none of stop conditions provided, training continues till method "stopTraining"
                                              // is called


      type: "back-propagation",               // default: "back-propagation"
      // backpropagation specific detailed options:

      learningRate: 0.3,    // scales with delta to effect training rate --> number between 0 and 1
      momentum: 0.1,        // scales with next layer's change value --> number between 0 and 1
    }
  */
  .then(trainingInfo => {
  });


ai.stopTraining() // stops training (if ongoing) and fires promise for ai.train


ai.verify(data, options) // data can be a binary stream or typed array
  /*
    example options object could look like that:
    {
      errorMeasureType: "mean-square",        // default: "mean-square"
    }
  */
  .then(verificationInfo => {
  });


ai.run(input, normalizeInput, denormalizeOutput) 
  // if normalizeInput callback not provided or null then input expected to be a typed array of dataType of NN, 
  // if denormalizeOutput not provided or null then typed array of dataType of NN should be returned

  .then(output => {
  })

ai.reset()  // Resets any internal state, historical data inside NN and
            // recurrent data
            // Reseting internal state is performed by assigning default
            // values to internal state type model properties


ai.getInternalState()
ai.setInternalState(state)
// Allows to store and restore state of NN for recurent and statefull NNs
// If state is null or undefined then equivalent to ai.reset()
  


ai.export({
  to: 'json' // expected type of export output ("json", "object" - predefined for "core" domain)
  /* ... other properties of given export type ... */
}).then(nnInExportedFormat => {});

/*
  options example for "json" and/or "object" type of export:
  {
    setupDataAs: "base64",  // available options: base64, array
    optimize: "minify",     // moves all possible parameters to setup.data, mangles all names, removes unused parts
                            // combines all NNs into one (if "net" operation used), removes properties that have and
                            // match default values
                            // if property not provided then returns in exact form as was provided
                            // if property is "pretty" moves all possible properties values from "setup" to layers 
                            // and pipes

    layers: {               // if provided - exports only selected layers/selected range of layers as NN JSON object
                            // allows to make autoencoder NNs or to split NNs so they could be runned in chain of 
                            // separate sub-NNs across multiple execution units
                            // should throw an error if there is any layer or pipe that would make splitting NN
                            // impossible
      from: "input",
      to: "output"
    }
  }
  returns always fully functional NN model in JSON string
*/


```

## Using a variety of hardware devices

 - API implementations should provide possibility of use different type of hardware devices in a system simultaneusly (all CPU cores, discrete and dedicated GPUs, DSPs, FPGAs) through promises API only
 - Javascript engine should decide on what execution unit NN tasks will be executed and it doesn't have to be same execution unit for all tasks


## Layers and pipes - advanced neural networks architectures

 - First layer and pipes located on first layer are always considered an "input" type of operation, unless specifically permitted, no other types can't be used on first layer
 - Pipes is a mechanism to introduce advanced NN architectures, their flexibly allow to get desired schema/shape of NN, some more sophisticated recurrent NN's can be easily achieved with it as well
 - Pipes can direct neuron outputs in both directions (forward and backward) - also to current layer/pipe as well
 - Piping to same or above layer means that outputs of that particular pipe/layer will be available there in next run
 - Activation function can be modified for every layer and every pipe in a layer
 - Pipes can also be assigned names
 - Layer can be build out of pipes only (as an option)
 - Pipes on first layer adds additional inputs in order of appearance
 - Pipes without property "to" pipe simply to next layer
 - Layer can also define property "to"
 - If multiple pipes pipe to same pipe end then all of their outputs are joined together in order of appearance in JSON model. If outputs are multidimmensional, then number of all dimmensions (except last one) must match (otherwise error thrown). Last dimmension for result of such a join is in this case sum of last dimmensions
 - If piping to fixed size pipe end then size of piping layer or pipe must equal to size of pipe end, otherwise error thrown

 Core properties summary:
  - **domain** property defines a domain for operations (??and activation functions??) in ml model, default: "core"
  - **name** property defines a name for layer or pipe, all names must be unique
  - **type** property defines a type of operator for given layer or pipe, available options for core ml: neurons, lstm-n, lstm-p, gru. Where "n" stands for "normalized", and "p" for "pseudo". Default is defined at domain level and can be changed at model level by providint "type" property in root of model
  - **activation** property defines an activation operation for operation in given layer or pipe. It expects operation with given name to be defined and have set "isActivation" property set to true
  - **connections** property defines a way that neurons from given pipe/layer are connected to predecessing layers/pipes
  - **pipes** property defines a list of pipes for given layer
  - **size** property defines how many inputs/neurons/outputs is in layer/pipe operation, can be a number or array - if array provided (eg [2, 3, 5]) then assumed multidimmensional output, single number is just a shorthand for: [number]
  - **to** property defines where to pipe neuron/operation outputs(or data inputs if on input layer) of current layer/pipe
  - **history** property defines additional outputs of current layer/pipe build out of values of neuron/operation outputs (or input data if on input layer) from given number of previous NN runs
  - **historyTo** property defines where to pipe previous NN runs values of neuron/operation outputs(or data inputs if on input layer) from current layer/pipe

```javascript
const ai = new WebAI.NeuralNetwork({
  dataType: 'u8',
  activation: "tanh", // activation operation for all layers and pipes (unless defined at layers or pipes)

  layers: [
    {
      name: "input", //optional - allows later to pipe by layer name
      size: 8
    },
    8, // regular layer with 8 neurons
    10,
    {
      size: 4,   // 4 outputs
      pipes: [    // multiple pipes posible
        {
          activation: "isrlu",    // overwrites default activation function (null will disable activation function at all)
          size: 16,              // adds 16 neurons to current layer of which outputs will be available at piped layers/pipes
          to: ['input', 1]        // piped to named layer (or pipe) and by index of layer
        }
      ]
    }
  ],

});

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
          size: 3,
          history: 2,           // reserves memory for 6 additional values (size * history = 2 * 3 = 6) 
                                // to keep two previous values of given 3 inputs
          historyTo: ['input']  // pipe historical values to input layer (will create hidden 6 inputs)
                                // note that original 3 inputs are still piped directly to next layer
        },
        {
          name: 'coordinates',
          size: 2
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

      size: 3      // 3 neurons of GRU LSTM
    },
    18, // regular layer with 18 neurons
    3   // 3 neurons output layer, last layer always is an output layer
        // equivalent to: {size: 3} and {type: "neurons", size: 3} and {connections: "all-to-all", type: "neurons", size: 3}
        //       and and {connections: "all-to-all", type: "neurons", activation: "tanh", size: 3}   :-)
  ],

});

```

## Simple example of combining bigger NN out of smaller NNs
```javascript

const smallerNN1 = {
  dataType: 'u8',
  activation: 'relu'
  layers: [8, 8, 8],
  setup: {
    data: '',
    instructions: ''
  }
}

const smallerNN2 = {
  dataType: 'u8',
  activation: 'relu'
  layers: [8, 8, 8],
  setup: {
    data: '',
    instructions: ''
  }
}


const ai = new WebAI.NeuralNetwork({
  dataType: 'u8',
  activation: 'relu'
  layers: [
    8,
    8,
    {
      type: "net",    // Reserved operation name for combining neural networks
                      // Can be used also at input layer

      net: smallerNN1 // Will throw error if different domains or data types
                      // "net" operation has its own names context for piping
    },
    {
      type: "net",
      net: smallerNN2
    }

  ],

});

```


## API for atomic ML operations

```javascript

WebAI.getOperations(operationsQuery)
  /*
    where operationsQuery could look like this:
    {
      domain: "operations_domain eg: 'core'"
      dataType: "fp16",
      variant: "default" // optional variant of operation, if skipped then "default" assumed
    }
  */

  .then(webAiOperations => {
    /*
      would return object with operation functions:
      {
        ...
        fusedMatMul(a, b, transposeA, transposeB, bias, activationFn) {},
        someStatefullOperation(a, b, state) {},
        ...
      }
    */
    const A = [/* ... */];
    const B = [/* ... */];

    const state1 = webAiOperations.someStatefullOperation.getNewState();
    const someProduct = webAiOperations.someStatefullOperation(A, B, state1);

    const result = webAiOperations.fusedMatMul(A, B, true, true, 0.0, "tanh");

  })

// defines a new domain
WebAI.defineDomain({
  domainName: "domain_name",      // throws error when domain already exist
  defaultDataType: "fp32",        // optional
  defaultOperationType: "neurons" // optional
});




// defines a new operation for given domain, operation will be assigned to "default" variant
WebAI.defineOperation( // throws error if operation already exist
  operationDescriptionObject,
  /*
    example of operationDescriptionObject:
    {
      name: "name_of_operation",
      domain: "some_existing_domain_name",
      dataType: "fp32",

      model: {
        isActivation: true,           // If property is defined and its value is true then operation can 
                                      // be used as activation function

        allowedOnFirstLayer: false,   // Throws error if someone tries to use such an operation
                                      // on first layer (inputs layer) if this set to false

        defineParams: ["*param1", "!param2", "!*param3", "paramN", paramObject, paramObject1, paramObjectN], 
                                                  // List of all properties of this operation available
                                                  // to JSON model
                                                  // Properties can be defined via shorthand or object
                                                  // (details below)
        
        operation: [":param1", ":param1", "param2", "paramN"],
                        // Allows to call operation_itself with given parameters 
                        // from model

        // it is possible to pass same param to function multiple times

      }

    }
  */

  function operation_itself(operation_params) {
    // Js code itself can make use of any existing API (including webgpu, webgl, webml)
    // Function should return typed array or array or nested arrays 
    // Multidimmensional nested arrays can be flatten to single dimmension
  },
);
```

### Operation description object - model.defineParams:

```javascript
// all boolean values by default are false and can be ommited

const paramObject = {
  name: "parameter_name",
  internalState: false,         // Property will be not available to model, only as internal
                                // state. Access to it is via "state" property or given name

  arrayType: true,              // Such a parameter can be a pipe end (accepts piped pipes 
                                // and layers) or array (also array of arrays for 
                                // multidimmensional arrays)

  noSetup: true,                // Prevents given parameter from being stored in setup.data
                                // (if noSetup is false and string data will be provided
                                // to property then it will be not stored in setup anyway)

  subjectOfTraining: true,      // Informs training algorithms that this property can be
                                // a subject of training

  trainOnlyIfNotInModel: true,  // Prevents training on property data if provided in model

  paramsForDefault: ["param1", "param2", ":param2", "&param2", "paramN"],   
                                // Allows to call "default" function with given parameters 
                                // from model


  default: (p1, p2, p3) => {    // If default property is a function then calculation is 
    return 10;                  // to retrieve default value
  }                             // In other case, default value is a property value
}
```

#### Array type arguments

  * can be assigned a string - then it means that it is a pipe end with given name
  * can be assigned a number - then it determines one dimmensional array with given number of elements that should be passed from setup.data to operation
  * can be assigned array or nested arrays - data are provided directly from model, dimmensions passed to operations are determined automatically based on input array
  * can be assigned an object with property "size" (eg. { size: [10, 20] } then it means that expected data for 2 dimmensional array should be placed in setup.data
  * if contains dimmensions data then they will be stored to setup.data

#### Shorthands

  * Properties used to define parameters that should be passed from model to called operation function (paramObject.paramsForDefault, model.operation) may contain strings which starts with:
    - ":" - property dimmensions instead of property value are passed - if such a property in a model is provided as an array then dimmensions are guessed from provided array (or array of arrays)
    - "&" - layer/pipe id of value producer or id of data supplied by model is passed. For automatically merged values (multiple layers/pipes and data from model to one input) all id's are provided.

  * Property "model.defineParams" is an array which may contain string literals of names or detailed objects describing parameters. If string provided then it is considered as shorthand for object. String can start with multiple special characters which are shorthand for:
    - "*" - equivalent to: paramObject.arrayType = true
    - "!" - equivalent to: paramObject.noSetup = true
    - "@" - equivalent to: paramObject.subjectOfTraining = true
    - "~" - equivalent to: paramObject.internalState = true

  Example:
 
  "!@*weights" is equivalent to: { name: "weights", noSetup: true, arrayType: true, subjectOfTraining: true } 


#### Reserved property/parameters names

  * "input" - array type parameter, a default pipe end
  * "activation" - reserved name for passing activation function to custom operation
  * "activationStr" is a reserved name for passing name of activation function
  * "state" is a reserved word for layer/pipe state, holds object with internal state parameters
  * "id" is a reserved word for layer/pipe id's and data id's (each has unique id assigned at object creation from JSON model) where pipe/layer ids are positive integers and data id's are negative integers
  * "size" is a reserved name for layer/pipe output dimmensions. If no "size" provided in model then defined default will be used ("size" property can be defined as paramObject, but can contain only properties that helps providing default values, other are forbidden). If no default, then it throws error (this is a guarantee that output dimmensions are known without running NN operations)

All above properties are predefined and don't have to be defined with operation definition (unless defaults have to be provided for "input" or "size", then these properties can be defined again)

Property names mentioned in "Core properties summary" can not be repurpoused (except property "connections") and are reserved too


#### Operation variants

Same operation might have different variants. Different operation variants might aim different purpouses. Predefined variants are: "default" and "standalone". First is used for default model compilation/running/training/veryfing, second allows access to operations programatically as regular functions. Binding to "default" variant can be obtained only via "WebAI.defaineOperation" method. Variants can be used for different purpouses, like: exporting to different programming languages or to different NN formats, running on non standard computation envirionment (CPLD, FPGA, DSP, remote computing resources) and so on ... Running, compilation, training and verifying function decides what variant of operation they'll use to operate and can use many of them at the same time.

```javascript
// defines a new operation variant for given domain
WebAI.defineOperationVariant( // throws error if variant already exist or operation doesn't exist
  operationVariantDescriptionObject,
  /*
    example of operationVariantDescriptionObject:
    {
      name: "name_of_operation",
      domain: "some_existing_domain_name",
      dataType: "fp32",
      variant: "variant_name",
      operation: [":param1", ":param1", "param2", "paramN"],
            // Allows to call operation_itself with given parameters from model
            // Only parameters defined in "default" variant of operation can
            //  be used
    }

  */

  function operation_itself(operation_params) {
  },
```

#### Other informations

  * All numeric and boolean params not provided specifically in model and not having default value will be placed in models property "setup.data" and initialized with random values
  * Booleans converted to numeric have values: false: 0, true: !=0
  * If array type parameter is not provided, then default from property definition will be computed. Default can return - numbers (meaning single dimmensional array with size of value of number), arrays (also nested) or object with property "size" containing array with dimmensions. In first and last case data for will be placed in models property "setup.data" and initialized with random values. In case of no default in property definition and parameter not provided at model then error should be thrown (can not determine array size).
  * All data in setup.data are stored in order of appearance in JSON model and then in order of appearance in model.defineParams. Any mismatch between expected amount of data in setup.data and actually available will cause error.
  * default values are not stored in setup.data (since no need for)


#### Reserved operation types names

  * input - for assigning inputs (first layer and all pipes at first layer by default have this type)
  * net   - for attaching another neural network into current structure

## Example with custom operations on model level

```javascript

WebAI.defineDomain({
  domainName: "custom-domain",
  defaultDataType: "fp32",
  defaultOperationType: "neurons"
});
WebAI.defineOperation(

  {
    name: "addIfAbove",
    domain: "custom-domain",
    dataType: "fp32",

    model: {
      defineParams: ["*a", "*b", "value",     // Array type parameter a and b, non array type parameter "value"
        {                                     // (notice no default input for this operation)

          name: "size",                       // We want to calculate output dimmensions, that is allowed use
                                              // of predefined reserved param, but it can only define 
                                              // properties to calculate default

          paramsForDefault: [":a", ":b"],     // ":" - get dimmensions of parameter a and b
          default: (lengthOfA, lengthOfB) => (lengthOfA[0] > lengthOfB[0]) ? lengthOfA : lengthOfB;
        }                                     // Notice that dimmensions are passed as array
                                              // Even single dimmensions passed as number will be converted to array with length "1"
      ],
      operation: ["a", "b", ":a", ":b", value],
    }
  },

  (a, b, a_length, b_length, valueToCompareWith) => {
    if (a_length[0] !== b_length[0]) throw new Error('inputs have to have same lengths')
    let result = [];
    for (let i = 0; i < a.length; i++) {
      if (a[i] > valueToCompareWith && b[i] > valueToCompareWith) {
        result[i] = a[i] + b[i];
      } else {
        result[i] = 0;
      }
    }
    return result;
  }
);
WebAI.defineOperation(
  {
    name: "onlyLowerThan",
    domain: "custom-domain",
    dataType: "fp32",
    model: {
      defineParams: ["*input", "value", 
        {                              
          name: "size", 
          paramsForDefault: [":input"],
          default: calc: length => length  // output with size of input
        }
      ],
      operation: ["input", "value", ":input"],
    }
  },
  (input, valueToCompareWith, input_length) => {
    let result = [];
    for (let i = 0; i < input_length[0]; i++) {
      if (input[i] < valueToCompareWith) {
        result[i] = input[i];
      } else {
        result[i] = 0;
      }
    }
    return result;
  }
);
WebAI.defineOperation(
  {
    name: "neurons",
    domain: "custom-domain",
    dataType: "fp32",
    model: {
      defineParams: ["activation", "connections",
        {
          name: "bias",
          subjectOfTraining: true
        },
        {
          name: "weights",
          arrayType: true,
          subjectOfTraining: true,
          paramsForDefault: [":input", "size", "connections"], // input and size are predefined properties
                                                                // so we can use them here without defining
          default: (inputsDimmensions, neuronsDimmensions, connectionsType) => {
            // for sake of simplicity skipped taking into consideration connection type here

            return { size: [...inputDimmensions, ...neuronDimmensions] };
                                      // object passed as default to array property means that
                                      // it is dimmensions object and based on it data will be
                                      // be stored in setup.data with random values at the begining
          };
        }
      ],
      operation: ["input", "weights", "bias", "activation", "connections", ":input"],
    }
  },
  (inputs, weights, bias, activationFunction, connectionsType, neuronsCount) => {
    let result = new Float32Array(neuronsCount.reduce((a, b) => a * b));
    // some code for neuron layers that calls activationFunction( ) for each neuron
    return result;
  }
);


//  =======================================================================================
//         And "voila" - this is how custom operations could be used from "model":
//  =======================================================================================


const ai = new WebAI.NeuralNetwork({
  domain: 'custom-domain',
  dataType: 'fp32',

  layers: [
    {
      name: "input",                  // could be skipped here, but placed it for readibility of code
      pipes: [
        {
          name: "A",                  // could be skipped here, but placed it for readibility of code
          size: 30,
          to: ['addIfAboveInputA'] 
        },
        {
          name: "B",                  // could be skipped here, but placed it for readibility of code
          size: 30
          to: ['addIfAboveInputB'] 
        }
      ]
    },
    {
      // operation here doesn't support default input, so naming this layer wouldn't allow pipes/layers to acces it by layer name
      type: 'addIfAbove',
      a: 'addIfAboveInputA',          // assigned unique name to parameter A of addIfAbove custom operation
      b: 'addIfAboveInputB',          // assigned unique name to parameter B of addIfAbove custom operation
      value: 0                        // if this property wouldn't be placed here its value would be expected in "setup.data" field of model
    },
    {
      type: 'onlyLowerThan',
      value: 50
    },
    30 // layer of "neuron" type operation with 30 neurons and connections type "every-to-every"
  ]

}); 


```


## Further things to keep in mind

There could be also designed standardized JSON format for simple normalizations/denormalizations of numbers and enumerations

Possibility to add custom training algorithms to domains

## Links
https://towardsdatascience.com/a-deeper-understanding-of-nnets-part-1-cnns-263a6e3ac61

https://becominghuman.ai/a-deeper-understanding-of-nnets-part-2-rnns-b32240998fa9

https://medium.com/@godricglow/a-deeper-understanding-of-nnets-part-3-lstm-and-gru-e557468acb04



https://www.wikiwand.com/en/Activation_function

https://webmachinelearning.github.io/

https://webmachinelearning.github.io/webnn/



## Credits

Inspired on Brain.js and ConvNetJS