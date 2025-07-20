Of course. Here is the clean, fully updated document without any annotations indicating the changes. It reads as a single, cohesive specification.

***

# WebAI
API proposal for web based neural networks/machine learning. This updated draft incorporates concepts from modern ML frameworks like automatic differentiation, graph compilation, and advanced data pipelines to ensure performance, flexibility, and future-readiness.

# Background
Two years ago I've joined W3C [webmachinelearning standardization group](https://www.w3.org/community/webmachinelearning/) hoping to influence the result API. This repository was supoused to be my input for discusstion of future shape of machine learning API.

Given below "Basic goals" started to "draw" this counter proposal but it didn't get into discussion at all (nor even just a goals). I was quickly explained that my voice doesn't matter, was told that it is up to browser vendors (aka google, microsoft) what they want (and what they don't) and what they will implement and these are the reasons why my proposals and goals are not gonna get into an agenda of discussion.

## Basic goals for web based machine learning

 - Providing simple to learn and use machine learning API
 - Addressing the needs of smaller companies, individuals, hobbyist and even kids who learn JavaScript programming
 - "descientifing" a bit machine learning (to have something a bit closer to what is a [brain.js](https://brain.js.org/#/))
 - Avoiding necessity of 3rd party libraries as it happen to webgl for example - cool API, but at the end relatively heavy Three.js is used in over 90% of cases
 - No ["lock in"](https://en.wikipedia.org/wiki/Vendor_lock-in) of standards offered and promoted by "big tech" companies
 - Allowing to train and run AI in web browsers, with primitives that enable advanced scenarios like federated learning.
 - Allowing to use of all possible and available hardware accelerations (simultaneusly) for training and running NNs - all installed gpu's (vulkan, opencl, glsl, cuda), dsp's, fgpa's, attached AI devices, cpu's
 - Achieving near-native performance through an optimizing graph compiler that performs tasks like operator fusion, memory planning, and dead-code elimination.
 - Neural networks architecture as flexible descriptive JSON format that can be:
    * easily maintained by developers
    * easily compiled by an optimizing graph compiler into high-performance executable code
    * easily transferred between proceeses or machines
    * easily converted into a code in another programming languague
    * flexible enough so existing models from tensorflow or caffe or other popular libraries could be easily converted
    * easily used by visual neural network designing/training tools
    * suitable for basic and for scientific purpouses


## API TLDR;

 * WebAI.getCapabilities()
 * WebAI.getOperations(operationsQuery)
 * WebAI.quantize(modelObject, options)

 * `const dataset = new WebAI.Dataset.from(source, options)`
 * `dataset.map(mapFn).shuffle(bufferSize).batch(batchSize).prefetch(count)`

 * WebAI.defineDomain(domainDescriptionObject);
 * WebAI.defineOperation(operationDescriptionObject, operation, gradientFunction)
 * WebAI.defineOperationVariant(operationVariantDescriptionObject, operation)
 * WebAI.defineExport(exportDescriptionObject, exportFunction)
 * WebAI.defineRun(runDescriptionObject, runFunction)
 * WebAI.defineTrain(trainDescriptionObject, trainFunction)
 * WebAI.defineVerify(verifyDescriptionObject, verifyFunction)

 * const ai = new WebAI.NeuralNetwork(nnModelObject)
 * ai.train(data, options) // `data` can be a WebAI.Dataset instance
 * ai.stopTraining()
 * ai.verify(data, options) // `data` can be a WebAI.Dataset instance
 * ai.run(input, options)
 * ai.runStream(input, options)
 * ai.reset()
 * ai.getInternalState()
 * ai.setInternalState(state)
 * ai.export(options)
 * ai.computeGradients(batchData)
 * ai.applyGradients(gradients)


## Simple self explanatory JavaScript example


```javascript

WebAI.getCapabilities(optionalCapabilitiesQueryObject)
  .then(capabilities => {
    /*
      capabilities object could look like that:
      {
          domains: {
            'core': ['fp16', 'fp32', 'fp64', 'u8', 'u16', 'u32', 'u64'],
            'http://w3c/tensorflow-2018.12': ['fp16', 'fp32', 'fp64', 'u8', 'u16', 'u32', 'u64'],
            'http://w3c/onnx-2019.06': ['fp16', 'fp32', 'fp64', 'u8', 'u16', 'u32', 'u64'],
          }
      }
    */
  });



const ai = new WebAI.NeuralNetwork({ // This declarative object is the input to an optimizing graph compiler.
  domain: 'core',
  dataType: 'fp16',
  minMops: 300,
  activation: "tanh",
  io: {
    input: { normalize: { type: "rescale-linear", from: [0, 255], to: [0, 1] } },
    output: { denormalize: { type: "one-hot-to-label", labels: ["cat", "dog"] } }
  },
  layers: [784, 128, 10], // e.g., for MNIST [input, hidden, output]
  setup: {
    data: '', // Base64 encoded weights and biases...
  }
});


// Modern Data Handling with WebAI.Dataset
// This API enables high-performance, asynchronous data loading that
// won't block the main thread or the GPU.

// Source data (can be a massive array, an iterator, or a URL)
const trainingDataSource = [
  { image: [...], label: 'cat' },
  { image: [...], label: 'dog' },
  /* ... thousands more items ... */
];

// Define a processing function.
// This is used to apply transformations like normalization or augmentation.
function preprocess(dataItem) {
    const normalizedImage = dataItem.image.map(pixel => pixel / 255.0);
    const oneHotLabel = dataItem.label === 'cat' ? [1, 0] : [0, 1];
    return { input: normalizedImage, output: oneHotLabel };
}

// Create a high-performance data pipeline
const trainingDataset = new WebAI.Dataset.from(trainingDataSource)
  .map(preprocess)      // Apply our preprocessing function to each item
  .shuffle(1000)        // Shuffle the data with a 1000-item buffer
  .batch(32)            // Group items into batches of 32
  .prefetch(2);         // Asynchronously prepare the next 2 batches while the current one is training


// Training uses the dataset directly.
// The WebAI engine will iterate over the dataset until a stop condition is met.
ai.train(trainingDataset, {
    maxIterations: 20000,
    errorThreshold: 0.05,
    learningRate: 0.3,
    momentum: 0.1,
  })
  .then(trainingInfo => {
    console.log('Training complete:', trainingInfo);
  });


ai.stopTraining() // stops training (if ongoing) and fires promise for ai.train


ai.verify(verificationDataset, { errorMeasureType: "mean-square" })
  .then(verificationInfo => { /* ... */ });


ai.run(input, { denormalize: false })
  .then(output => { /* ... */ });

// Method for streaming inference, perfect for LLMs
for await (const partialResult of ai.runStream(input)) {
  // e.g., partialResult could be { token: '...' }
}

ai.reset()
ai.getInternalState()
ai.setInternalState(state)
ai.export({ to: 'json', optimize: 'minify' }).then(exportedModel => {});


// Primitives for advanced scenarios like Federated Learning
// These low-level methods are powered by the built-in autodiff engine.
const gradients = await ai.computeGradients(batchData);
// `gradients` can now be sent to a server for aggregation.
ai.applyGradients(aggregatedGradients);
```

## Using a variety of hardware devices

 - API implementations should provide possibility of use different type of hardware devices in a system simultaneusly (all CPU cores, discrete and dedicated GPUs, DSPs, FPGAs) through promises API only.
 - The internal graph compiler is responsible for analyzing the neural network graph and the available hardware. It will partition the graph and schedule operations on the most suitable execution units to maximize throughput.


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
  - **domain** property defines a domain for operations in ml model, default: "core"
  - **name** property defines a name for layer or pipe, all names must be unique
  - **type** property defines a type of operator for given layer or pipe.
  - **activation** property defines an activation operation for the operation in given layer or pipe.
  - **pipes** property defines a list of pipes for given layer
  - **size** property defines how many inputs/neurons/outputs is in layer/pipe operation. To handle dynamic shapes, a dimension can be set to `null` (e.g. `[null, null, 3]`).
  - **shapeHints** optional property to provide optimization hints for dynamic shapes.
  - **to** property defines where to pipe neuron/operation outputs of current layer/pipe
  - **history** property defines additional outputs of current layer/pipe build out of values from previous NN runs
  - **historyTo** property defines where to pipe historical values.

The declarative nature of layers and pipes, including the powerful `iterator` type, provides the graph compiler with the high-level semantic information needed to perform advanced optimizations, such as loop unrolling or state management, that would be impossible with imperative code.

```javascript
const ai = new WebAI.NeuralNetwork({
  dataType: 'fp32',
  activation: "tanh",
  layers: [
    {
      name: 'input',
      size: 3,
      history: 2,
      historyTo: ['input']
    },
    {
      type: "gru",
      size: 3
    },
    18,
    3
  ],
});
```

### Example: Diffusion Model with `iterator` Control Flow
To support complex, iterative processes without inefficient JavaScript loops, the special `iterator` layer type provides a declarative hint to the graph compiler.

```javascript
const diffusionModel = {
  layers: [
    {
      type: "iterator",
      steps: { from: 999, to: 0, step: -1 }, // Loop definition
      state: [ // State variables carried between iterations
        { name: "x", pipeFrom: "initial_noise" }
      ],
      net: { // The sub-network executed in each step
        // The compiler understands this `net` will be executed multiple times
        // and can create a highly optimized, single-kernel execution plan.
        layers: [
          { name: "x_input", pipeFrom: "x" },
          { name: "t_input", pipeFrom: "iterator:step" },
          { type: "net", net: uNetModel, to: "predicted_noise" },
          { type: "ddpmUpdateStep", to: "x" }
        ]
      }
    }
  ]
};
```

## API for atomic ML operations

This is the extensibility API that makes WebAI future-proof. It is built upon a core **Automatic Differentiation (Autodiff)** engine. This means developers can define the forward pass of a new operation, and the system can automatically compute the gradients required for training, dramatically simplifying the creation of new, trainable layers.

```javascript
WebAI.defineTrain(
  { name: "AdamW", domain: "core" },
  // A training function is now an "optimizer". It defines the training loop
  // and uses the core `computeGradients` and `applyGradients` methods.
  // It no longer needs to implement backpropagation itself.
  (internalModel, trainingData, options, controlSignal) => {
    // Loop over the dataset
    for await (const batch of trainingData) {
      // 1. Compute gradients using the built-in autodiff engine
      const gradients = await internalModel.computeGradients(batch);

      // 2. Apply optimizer logic (e.g., AdamW update rule) to the gradients
      const updatedGradients = applyAdamWLogic(gradients, options);

      // 3. Apply the updated gradients to the model's weights
      await internalModel.applyGradients(updatedGradients);

      // Check for stop signals, report progress, etc.
    }
    return Promise.resolve({ finalError: /*...*/ });
  }
);


// defineOperation now includes an optional gradient function.
WebAI.defineOperation(
  operationDescriptionObject, // Contains metadata like name, domain, params
  function operation_itself(operation_params) {
    // The forward pass logic.
  },
  // Optional: The backward pass (gradient) function.
  // For most operations composed of existing primitives (math, etc.), the autodiff
  // engine can derive this automatically. You only provide this for complex,
  // atomic operations or for performance-optimized custom gradients.
  function gradient_function(forwardInputs, output, outputGradient) {
    // Logic to compute the gradient of the loss with respect to the inputs.
    const { x, y } = forwardInputs;
    return {
      x: outputGradient, // Gradient for input 'x'
      y: outputGradient  // Gradient for input 'y'
    };
  }
);
```

### Example with custom operations on model level
```javascript
// Defining a custom "neurons" layer.
WebAI.defineOperation(
  {
    name: "neurons",
    domain: "custom-domain",
    model: {
        defineParams: ["*input", "@*weights", "@bias", "activation"], // @ marks params as trainable
        /* ... other params ... */
    }
  },
  (inputs, weights, bias, activationFunction) => {
    // Forward pass: a simple matrix multiplication, bias add, and activation.
    // This is all the developer needs to write for the forward pass.
    const output = activationFunction(matrixMultiply(inputs, weights) + bias);
    return output;
  }
  // NO gradient function needed here!
  // The WebAI autodiff engine understands how to differentiate matrix multiplication,
  // addition, and standard activation functions. It will automatically construct
  // the backward pass for this operation by chaining the gradients of its
  // constituent parts.
);

// The model definition using this custom operation remains unchanged.
const ai = new WebAI.NeuralNetwork({
  domain: 'custom-domain',
  dataType: 'fp32',
  layers: [
    { name: "input", size: 30 },
    { type: 'neurons', size: 15 },
    { type: 'neurons', size: 30 }
  ]
});
```

## Further things to keep in mind

- **Automatic Differentiation Engine:** The core of WebAI's trainability. It automatically calculates gradients for complex models, freeing developers from implementing backpropagation manually. New operations defined with `defineOperation` become automatically differentiable if they are composed of existing primitives.
- **Optimizing Graph Compiler:** Before execution, the declarative JSON model is compiled into a high-performance computation graph. This compiler performs optimizations like **operator fusion** (merging multiple layers into one hardware operation), **memory planning** (reusing memory buffers), and targeting specific hardware accelerators, resulting in performance far exceeding simple interpreters.
- **Advanced Data Pipelines:** The `WebAI.Dataset` API is the standard for handling data. It provides a fluent, chainable interface for building efficient, asynchronous, and out-of-core data loading pipelines, preventing data I/O from becoming a bottleneck during training.
- **Primitives for Federated & Distributed Learning:** The low-level `ai.computeGradients` and `ai.applyGradients` methods are not just for custom training loops; they are the fundamental building blocks for privacy-preserving federated learning, where gradients—not raw data—are shared and aggregated across clients.
- **Standardization of Primitives:** For this API to gain widespread adoption, it is crucial that a "standard library" of operations (`conv2d`, `layerNorm`, `selfAttention`, `graphConv`, etc.) is provided in `core` domains.
- **Debugging and XAI:** The API should be extended with debugging hooks. `ai.run({ debug: true })` could return intermediate activations for easier debugging.
- **Quantization:** A helper utility `WebAI.quantize(modelObject, options)` can handle Post-Training Quantization (PTQ). Quantization-Aware Training (QAT) can be enabled with an option in `ai.train({ quantization: { mode: 'qat' } })`.
- **Security:** As models and operations can be defined in JavaScript, the execution environment must be robustly sandboxed to prevent malicious code from accessing sensitive system resources beyond the intended scope of the API.

## Links
https://towardsdatascience.com/a-deeper-understanding-of-nnets-part-1-cnns-263a6e3ac61

https://becominghuman.ai/a-deeper-understanding-of-nnets-part-2-rnns-b32240998fa9

https://medium.com/@godricglow/a-deeper-understanding-of-nnets-part-3-lstm-and-gru-e557468acb04

https://www.wikiwand.com/en/Activation_function

https://webmachinelearning.github.io/

https://webmachinelearning.github.io/webnn/

## Credits

Inspired on Brain.js and ConvNetJS
