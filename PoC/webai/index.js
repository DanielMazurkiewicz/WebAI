const domains = {};


const defineCustomDomain({domainName, defaultDataType, defaultOperationType}) => {
  if (domains[domainName]) throw new Error(`Domain ${domainName} already exist`);
  domains[domainName] = {
    defaults: { dataType: defaultDataType, type: defaultOperationType },
    dataTypes: {}
  }
};



const defineCustomOperation({name, domain, dataType, initState, model}, operation) => {
  const domainRoot = domains[domain];
  if (!domainRoot) throw new Error(`Domain ${domain} doesn't exist`);

  let domainDetails = domainsRoot.dataTypes[dataType];
  if (!domainDetails) {
    domainDetails = domainsRoot.dataTypes[dataType] = {
      operationVariants: {
        "default": {
          [name]: {
            model,
            operation
          }
        }
      },
      runVariants: {}
    }
    return;
  } else if (domainDetails.operationVariants.default[name]) {
    throw new Error(`Variant "default" of operation ${name} (${dataType}) already exist in domain ${domain}`);
  }

  domainDetails.operationVariants.default[name] = {
    model,
    operation
  }

};



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




module.exports = {
  getCapabilities,
  getOperations,

  defineDomain,
  defineOperation,
  defineOperationVariant,
  NeuralNetwork
}