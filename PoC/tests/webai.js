const WebAI = require('../webai/index');

const domain = "webai";
const dataType = "fp32";

WebAI.defineDomain({
  domainName: domain,
  defaultDataType: dataType,
  defaultOperationType: "neurons"
});


/*WebAI.defineOperation({
  domain, dataType,
  name: "history",

  model: {
    defineParams: [{
      name: "size",
      callDefaultWith: [":input"],
      default: dimmensions => dimmensions
    }],

    callWith: ["input", "$input", "output"]
  }
}, (input, inputSize, output) => {
  var i = 0;
  for (;i < inputSize; i++) {
    output[i] = Math.tanh(input[i]);
  }
});*/


WebAI.defineOperation({
  domain, dataType,
  name: "tanh",

  model: {
    isActivation: true,

    defineParams: [{
      name: "size",
      callDefaultWith: [":input"],
      default: dimmensions => dimmensions
    }],

    callWith: ["input", "$input", "^input", "output", "^output"]
  }
}, (input, inputSize, inputOffset, output, outputOffset) => {
  var i = 0;
  for (;i < inputSize; i++) {
    output[outputOffset++] = Math.tanh(input[inputOffset++]);
  }
});


WebAI.defineOperation({
  domain, dataType,
  name: "neurons",

  model: {
    defineParams: [{
      name: "weights",
      arrayType: true,
      subjectOfTraining: true,
      callDefaultWith: ["$input", "#size"],
      default: (sizeOfInput, numberOfNeurons) => { size: [sizeOfInput * numberOfNeurons] }
    }, {
      name: "size", // if size property skipped then it assumes as many neurons as inputs
      callDefaultWith: [":input"],
      default: dimmensions => dimmensions
    }],

    callWith: ["input", "$input", "^input", 
                "#size", 
                "weights", "^weights", 
                "output", "^output"]
  }
}, (input, inputSize, inputOffset, 
    numberOfNeurons, 
    weights, weightsOffset, 
    output, outputOffset) => {

  var o = 0, i, result;
  for (;o < numberOfNeurons; i++) {
    result = 0;
    for (i = 0, inputPosition = inputOffset; i < inputSize; i++) {
      result += input[inputPosition++] * weights[weightsOffset++];
    }
    output[outputOffset++] = result / inputSize;
  }
});


WebAI.printDomainObject();
