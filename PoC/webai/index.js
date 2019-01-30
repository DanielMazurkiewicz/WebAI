const domains = {};

const defineDomain = ({domainName, defaultDataType, defaultOperationType, defaultActivationType}) => {
  if (domains[domainName]) throw new Error(`Domain ${domainName} already exist`);
  domains[domainName] = {
    defaults: { dataType: defaultDataType, type: defaultOperationType, activation: defaultActivationType },
    dataTypes: {}
  }
};
const {defineOperation, defineOperationVariant} = require('./defineOperation')(domains);



const getCapabilities = () => {};
const getOperations = (operationsQuery) => {};




class NeuralNetwork {
  constructor(model) {
    /*
      Precompile model: 
        * calculate all inputs and outputs sizes, 
        * assign id's 
        * unwrap activation operations
        * unwrap all shorthands
    */
  }
  prepareData(normalizeInput, normalizeOutput, arrayOfData) {}
  train(data, options) {}
  stopTraining() {}
  verify(data, options) {}
  run(input, normalizeInput, denormalizeOutput) {}
  reset() {}
  getInternalState() {}
  setInternalState(state) {}
  export(options) {}
}


const printDomainObject = () => {
  console.log(JSON.stringify(domains, null, 3));
}

module.exports = {
  getCapabilities,
  getOperations,

  defineDomain,
  defineOperation,
  defineOperationVariant,
  NeuralNetwork,


  printDomainObject
}