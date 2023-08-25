source ../config.env

az storage blob upload-batch \
    --source . \
    --destination $BLOB_STORAGE_CONTAINER \
    --account-name $BLOB_STORAGE_ACCOUNT \
    --pattern "mnist_images/subset/*" \
    --max-connections 10

az ml data create -f mnist-sample-dataset.yaml --workspace-name $WORKSPACE --resource-group $RESOURCE_GROUP