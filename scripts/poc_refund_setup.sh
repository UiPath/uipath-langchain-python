#!/bin/bash
# =============================================================================
# Ontology Write POC — staging setup
#
# Creates the contact-center refund entities on the logged-in Data Fabric
# tenant, seeds one refund scenario, and emits three artifacts into OUT:
#   - refund_entity_set.json  (DataFabricEntityItem list; referenceKey = GUID)
#   - refund_ontology.ttl       (OWL with df:entityKey matching the real names)
#   - refund_ids.env            (seeded record ids + entity ids for teardown)
#
# Status fields are plain STRING (the choice-set-values endpoint is currently
# unreliable on staging); the ontology still models OrderStatus as a StateField.
#
# Prereq: `uip login` to the target tenant.
# Usage:  bash scripts/poc_refund_setup.sh [OUT_DIR]   (default: ./poc_out)
# Teardown: bash scripts/poc_refund_teardown.sh [OUT_DIR]
# =============================================================================
set -eo pipefail
OUT="${1:-./poc_out}"
mkdir -p "$OUT"
P="Rfnd$(date +%H%M%S)"
idof() { python3 -c "import json,sys; print(json.load(sys.stdin)['Data']['Id'])"; }

echo "=== Entities (prefix $P) ==="
CUST_E=$(uip df entities create "${P}_Customer" --body "{\"displayName\":\"${P} Customer\",\"fields\":[{\"fieldName\":\"CustomerName\",\"type\":\"STRING\",\"isRequired\":true,\"lengthLimit\":200},{\"fieldName\":\"AccountTier\",\"type\":\"STRING\",\"lengthLimit\":50}]}" --output json 2>/dev/null | idof)
ORD_E=$(uip df entities create "${P}_PurchaseOrder" --body "{\"displayName\":\"${P} Order\",\"fields\":[{\"fieldName\":\"OrderNumber\",\"type\":\"STRING\",\"isRequired\":true,\"lengthLimit\":50},{\"fieldName\":\"TotalAmount\",\"type\":\"DECIMAL\",\"decimalPrecision\":2},{\"fieldName\":\"OrderStatus\",\"type\":\"STRING\",\"lengthLimit\":50}]}" --output json 2>/dev/null | idof)
RISK_E=$(uip df entities create "${P}_CustomerRisk" --body "{\"displayName\":\"${P} Risk\",\"fields\":[{\"fieldName\":\"RiskScore\",\"type\":\"INTEGER\"},{\"fieldName\":\"LifetimeValue\",\"type\":\"DECIMAL\",\"decimalPrecision\":2}]}" --output json 2>/dev/null | idof)
CONT_E=$(uip df entities create "${P}_Contact" --body "{\"displayName\":\"${P} Contact\",\"fields\":[{\"fieldName\":\"ContactReason\",\"type\":\"STRING\",\"lengthLimit\":50},{\"fieldName\":\"RefundAmount\",\"type\":\"DECIMAL\",\"decimalPrecision\":2},{\"fieldName\":\"OrderRef\",\"type\":\"STRING\",\"lengthLimit\":100},{\"fieldName\":\"Resolution\",\"type\":\"STRING\",\"lengthLimit\":50}]}" --output json 2>/dev/null | idof)
RFND_E=$(uip df entities create "${P}_RefundRequest" --body "{\"displayName\":\"${P} Refund\",\"fields\":[{\"fieldName\":\"ApprovedAmount\",\"type\":\"DECIMAL\",\"decimalPrecision\":2,\"isRequired\":true},{\"fieldName\":\"Reason\",\"type\":\"STRING\",\"isRequired\":true,\"lengthLimit\":500},{\"fieldName\":\"OrderRef\",\"type\":\"STRING\",\"lengthLimit\":100},{\"fieldName\":\"CustomerRef\",\"type\":\"STRING\",\"lengthLimit\":100},{\"fieldName\":\"RefundStatus\",\"type\":\"STRING\",\"lengthLimit\":50}]}" --output json 2>/dev/null | idof)

echo "=== Seed records ==="
CUST_R=$(uip df records insert "$CUST_E" --body '{"CustomerName":"Sarah Chen","AccountTier":"Gold"}' --output json 2>/dev/null | idof)
ORD_R=$(uip df records insert "$ORD_E" --body '{"OrderNumber":"ORD001","TotalAmount":200.00,"OrderStatus":"Delivered"}' --output json 2>/dev/null | idof)
RISK_R=$(uip df records insert "$RISK_E" --body '{"RiskScore":2,"LifetimeValue":5000.00}' --output json 2>/dev/null | idof)
CONT_R=$(uip df records insert "$CONT_E" --body "{\"ContactReason\":\"Refund\",\"RefundAmount\":200.00,\"OrderRef\":\"$ORD_R\",\"Resolution\":\"Open\"}" --output json 2>/dev/null | idof)

F="00000000-0000-0000-0000-000000000000"
# referenceKey = entity GUID — resolve_entity_set_async looks up by this value,
# and the CRUD endpoints require the id, not the entity name.
cat > "$OUT/refund_entity_set.json" <<JSON
[
  {"id":"$CUST_E","name":"${P}_Customer","folderId":"$F","referenceKey":"$CUST_E","description":"Customer master (read-only)"},
  {"id":"$CONT_E","name":"${P}_Contact","folderId":"$F","referenceKey":"$CONT_E","description":"Inbound contact / refund request"},
  {"id":"$ORD_E","name":"${P}_PurchaseOrder","folderId":"$F","referenceKey":"$ORD_E","description":"Order records"},
  {"id":"$RISK_E","name":"${P}_CustomerRisk","folderId":"$F","referenceKey":"$RISK_E","description":"Customer risk profile"},
  {"id":"$RFND_E","name":"${P}_RefundRequest","folderId":"$F","referenceKey":"$RFND_E","description":"Refund records"}
]
JSON

cat > "$OUT/refund_ontology.ttl" <<TTL
@prefix df:   <https://ontology.uipath.com/datafabric#> .
@prefix ex:   <https://ontology.example.com/refund#> .
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

ex:Customer a owl:Class ; rdfs:subClassOf df:ReadableEntity ; df:entityKey "${P}_Customer" .
ex:Contact a owl:Class ; rdfs:subClassOf df:WritableEntity ; df:entityKey "${P}_Contact" ; df:allowsOperation "update" .
ex:Order a owl:Class ; rdfs:subClassOf df:WritableEntity ; df:entityKey "${P}_PurchaseOrder" ; df:allowsOperation "update" .
ex:Risk a owl:Class ; rdfs:subClassOf df:WritableEntity ; df:entityKey "${P}_CustomerRisk" ; df:allowsOperation "update" .
ex:Refund a owl:Class ; rdfs:subClassOf df:WritableEntity ; df:entityKey "${P}_RefundRequest" ; df:allowsOperation "insert" .

ex:field_Order_Status a df:StateField ; df:fieldKey "OrderStatus" ; df:choiceSetKey "OrderStatusValues" .
ex:field_Risk_Score a df:MeasureField ; df:fieldKey "RiskScore" ; df:measureSemantics "additive" .
ex:field_Risk_LTV a df:MeasureField ; df:fieldKey "LifetimeValue" ; df:measureSemantics "additive" .
ex:field_Refund_Order a df:ReferenceField ; df:fieldKey "OrderRef" ; df:referencesEntity ex:Order .
ex:field_Refund_Customer a df:ReferenceField ; df:fieldKey "CustomerRef" ; df:referencesEntity ex:Customer .

ex:Order df:hasField ex:field_Order_Status .
ex:Risk df:hasField ex:field_Risk_Score , ex:field_Risk_LTV .
ex:Refund df:hasField ex:field_Refund_Order , ex:field_Refund_Customer .

ex:CreateRefund a df:InsertAction ; df:writeOperation "insert" ; df:targetEntity ex:Refund ; df:requiresHITL false .
ex:UpdateOrder a df:UpdateAction ; df:writeOperation "update" ; df:targetEntity ex:Order ; df:requiresHITL false .
ex:UpdateRisk a df:UpdateAction ; df:writeOperation "update" ; df:targetEntity ex:Risk ; df:requiresHITL false .
ex:UpdateContact a df:UpdateAction ; df:writeOperation "update" ; df:targetEntity ex:Contact ; df:requiresHITL false .

ex:Refund df:hasAction ex:CreateRefund .
ex:Order df:hasAction ex:UpdateOrder .
ex:Risk df:hasAction ex:UpdateRisk .
ex:Contact df:hasAction ex:UpdateContact .

ex:Contact df:relatedEntity ex:Customer , ex:Order .
ex:Refund df:relatedEntity ex:Order , ex:Customer .
ex:Risk df:relatedEntity ex:Customer .
TTL

cat > "$OUT/refund_ids.env" <<ENV
CONTACT_ID=$CONT_R
ORDER_ID=$ORD_R
RISK_ID=$RISK_R
CUSTOMER_ID=$CUST_R
E_CUSTOMER=$CUST_E
E_CONTACT=$CONT_E
E_ORDER=$ORD_E
E_RISK=$RISK_E
E_REFUND=$RFND_E
ENV

echo "=== DONE — artifacts in $OUT ==="
echo "Contact=$CONT_R Order=$ORD_R Risk=$RISK_R Customer=$CUST_R"
