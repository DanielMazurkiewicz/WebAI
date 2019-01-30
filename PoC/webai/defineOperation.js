module.exports = domains => {

  const shorthandParamDefinition = {
    "*": paramObject => paramObject.arrayType = true,
    "!": paramObject => paramObject.noSetup = true,
    "@": paramObject => paramObject.subjectOfTraining = true,
    "~": paramObject => paramObject.internalState = true,
  }


  const shorthandCallWith = {
    ":": callWithParamObject => callWithParamObject.kind = 'id',          //
    "&": callWithParamObject => callWithParamObject.kind = 'dimmensions', // dimmensions of given property passed as array
    "$": callWithParamObject => callWithParamObject.kind = 'size',        // flattened size - all dimmensions of given property multiplied together
    "#": callWithParamObject => callWithParamObject.kind = 'mulArrEl',    // multiply array elements and return result
    default: callWithParamObject => callWithParamObject.kind = 'value',
  }


  const unwrapShorthandCallWith = (paramName, callWithObject = {}) => {
    const unwrapFunction = shorthandCallWith[paramName[0]];
    if (unwrapFunction) {
      unwrapFunction(callWithObject);
      callWithObject.name = paramName.substring(1);
    } else {
      shorthandCallWith.default(callWithObject);
      callWithObject.name = paramName;
    }
    return callWithObject;
  }

  const unwrapCallWith = callWith => {
    if (!callWith) return;
    return callWith.map(c => {
      if (typeof c === 'string') {
        return unwrapShorthandCallWith(c);
      } else {
        return unwrapShorthandCallWith(c.name, JSON.parse(JSON.stringify(c)));
      }
    })
  }



  const unwrapShorthandParamDefinition = (paramName, paramObject = {}) => {
    let unwrapFunction, i;
    for (i = 0; unwrapFunction = shorthandParamDefinition[paramName[i]]; i++) {
      unwrapFunction(paramObject);
    }
    paramObject.name = paramName.substring(i);
    return paramObject;
  }

  const unwrapModel = model => { // !!not "immutable" function!!
    const {defineParams} = model;
    const result = {};
    let name;

    if (defineParams instanceof Array) {
      defineParams.forEach(param => {
        let paramObj;
        if (typeof param === 'string') {
          paramObj = unwrapShorthandParamDefinition(param);
        } else if (param.name) { // object
          paramObj = unwrapShorthandParamDefinition(param.name, JSON.parse(JSON.stringify(param)));
          paramObj.callDefaultWith = unwrapCallWith(paramObj.callDefaultWith);
        } else {
          throw new Error(`Improper model parameter definition:\n${JSON.stringify(param, null, 3)}`)
        }

        if (result[paramObj.name]) throw new Error(`Parameter ${paramObj.name} in parameters definition:\n${JSON.stringify(param, null, 3)}`)
        result[paramObj.name] = paramObj;
      });
    } else {
      for (let paramName in defineParams) {
        const paramObj = unwrapShorthandParamDefinition(paramName, JSON.parse(JSON.stringify(defineParams[paramName])));
        result[paramObj.name] = paramObj;
      }
    }

    model.defineParams = result;
  }




  const defineOperation = ({name, domain, dataType, model}, operation) => {
    const {callWith, ...modelObj} = model;

    const domainsRoot = domains[domain];
    if (!domainsRoot) throw new Error(`Domain ${domain} doesn't exist`);

    let domainDetails = domainsRoot.dataTypes[dataType];
    if (!domainDetails) {
      domainDetails = domainsRoot.dataTypes[dataType] = { operations: {} }
    }

    let operationsDetails = domainDetails.operations;
    if (!operationsDetails) {
      operationsDetails = domainDetails.operations = {
        [name]: {
          model: unwrapModel(modelObj),
          variants: {
            default: { // variant default
              callWith: unwrapCallWith(callWith),
              operation
            }
          }
        }
      };
    } else if (operationsDetails[name]) {
      throw new Error(`Variant "default" of operation (${dataType})${name} already exist in domain ${domain}`);
    }


    operationsDetails[name] = {
      model: unwrapModel(modelObj),
      variants: {
        default: {
          callWith: unwrapCallWith(callWith),
          operation
        }
      }
    }
  };


  const defineOperationVariant = ({name, domain, dataType, variant, callWith}, operation) => {

    const domainsRoot = domains[domain];
    if (!domainsRoot) throw new Error(`Domain ${domain} doesn't exist`);

    let domainDetails = domainsRoot.dataTypes[dataType], op;
    if (!domainDetails || !(op = domainDetails.operations[name])) 
        throw new Error(`Domain ${domain} doesn't have defined operation (${dataType})${name}`);
     

    if (operation.variants[variant])
        throw new Error(`Domain ${domain} have already defined variant ${variant} for operation (${dataType})${name}`);

    operation.variants[variant] = {
      callWith: unwrapCallWith(callWith),
      operation
    }
 };



  return {defineOperation, defineOperationVariant}
}