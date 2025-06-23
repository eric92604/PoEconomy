#!/bin/bash

# AWS PoEconomy Deployment Script - DynamoDB + S3
# This script deploys the PoEconomy architecture with optimized performance

set -e

# Configuration
STACK_NAME="poeconomy"
REGION="${AWS_DEFAULT_REGION:-us-east-1}"
PROFILE="${AWS_PROFILE:-default}"
CLOUDFLARE_TOKEN="${CLOUDFLARE_API_TOKEN:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 PoEconomy Architecture Deployment${NC}"
echo "=================================================="

# Check prerequisites
echo -e "${YELLOW}📋 Checking prerequisites...${NC}"

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    echo -e "${RED}❌ AWS CLI not found. Please install AWS CLI first.${NC}"
    exit 1
fi

# Check AWS credentials
if ! aws sts get-caller-identity --profile $PROFILE &> /dev/null; then
    echo -e "${RED}❌ AWS credentials not configured. Please run 'aws configure'.${NC}"
    exit 1
fi

# Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --profile $PROFILE --query Account --output text)
echo -e "${GREEN}✅ AWS Account ID: $ACCOUNT_ID${NC}"

# Check if stack exists
STACK_EXISTS=false
if aws cloudformation describe-stacks --stack-name $STACK_NAME --profile $PROFILE --region $REGION &> /dev/null; then
    STACK_EXISTS=true
    echo -e "${YELLOW}⚠️  Stack $STACK_NAME already exists. Will update.${NC}"
else
    echo -e "${GREEN}✅ Creating new stack: $STACK_NAME${NC}"
fi

# Validate CloudFormation template
echo -e "${YELLOW}🔍 Validating CloudFormation template...${NC}"
aws cloudformation validate-template \
    --template-body file://aws/cloudformation/poeconomy-infrastructure.yaml \
    --profile $PROFILE \
    --region $REGION > /dev/null

echo -e "${GREEN}✅ Template validation passed${NC}"

# Create Lambda deployment packages
echo -e "${YELLOW}📦 Creating Lambda deployment packages...${NC}"

# Create temp directory
TEMP_DIR=$(mktemp -d)
echo "Using temp directory: $TEMP_DIR"

# Package data ingestion Lambda
echo "Packaging data ingestion Lambda..."
mkdir -p $TEMP_DIR/data-ingestion
cp aws/lambda/data_ingestion_handler.py $TEMP_DIR/data-ingestion/index.py
cd $TEMP_DIR/data-ingestion
zip -r ../data-ingestion.zip . > /dev/null
cd - > /dev/null

# Package ML computation Lambda
echo "Packaging ML computation Lambda..."
mkdir -p $TEMP_DIR/ml-computation
cp aws/lambda/ml_computation_handler.py $TEMP_DIR/ml-computation/index.py
cd $TEMP_DIR/ml-computation
zip -r ../ml-computation.zip . > /dev/null
cd - > /dev/null

# Package API serving Lambda
echo "Packaging API serving Lambda..."
mkdir -p $TEMP_DIR/api-serving
cp aws/lambda/api_serving_handler.py $TEMP_DIR/api-serving/index.py
cd $TEMP_DIR/api-serving
zip -r ../api-serving.zip . > /dev/null
cd - > /dev/null

echo -e "${GREEN}✅ Lambda packages created${NC}"

# Deploy CloudFormation stack
echo -e "${YELLOW}🚀 Deploying CloudFormation stack...${NC}"

PARAMETERS="ParameterKey=EnvironmentName,ParameterValue=poeconomy"
if [ ! -z "$CLOUDFLARE_TOKEN" ]; then
    PARAMETERS="$PARAMETERS ParameterKey=CloudflareApiToken,ParameterValue=$CLOUDFLARE_TOKEN"
fi

if [ "$STACK_EXISTS" = true ]; then
    # Update existing stack
    aws cloudformation update-stack \
        --stack-name $STACK_NAME \
        --template-body file://aws/cloudformation/poeconomy-infrastructure.yaml \
        --parameters $PARAMETERS \
        --capabilities CAPABILITY_IAM \
        --profile $PROFILE \
        --region $REGION
    
    echo -e "${YELLOW}⏳ Waiting for stack update to complete...${NC}"
    aws cloudformation wait stack-update-complete \
        --stack-name $STACK_NAME \
        --profile $PROFILE \
        --region $REGION
else
    # Create new stack
    aws cloudformation create-stack \
        --stack-name $STACK_NAME \
        --template-body file://aws/cloudformation/poeconomy-infrastructure.yaml \
        --parameters $PARAMETERS \
        --capabilities CAPABILITY_IAM \
        --tags Key=Project,Value=PoEconomy Key=Environment,Value=Production Key=Architecture,Value=DynamoDB-S3 \
        --profile $PROFILE \
        --region $REGION
    
    echo -e "${YELLOW}⏳ Waiting for stack creation to complete...${NC}"
    aws cloudformation wait stack-create-complete \
        --stack-name $STACK_NAME \
        --profile $PROFILE \
        --region $REGION
fi

echo -e "${GREEN}✅ CloudFormation stack deployed successfully${NC}"

# Get stack outputs
echo -e "${YELLOW}📊 Getting stack outputs...${NC}"
OUTPUTS=$(aws cloudformation describe-stacks \
    --stack-name $STACK_NAME \
    --profile $PROFILE \
    --region $REGION \
    --query 'Stacks[0].Outputs')

API_URL=$(echo $OUTPUTS | jq -r '.[] | select(.OutputKey=="ApiUrl") | .OutputValue')
BUCKET_NAME=$(echo $OUTPUTS | jq -r '.[] | select(.OutputKey=="DataLakeBucket") | .OutputValue')

# Update Lambda function codes
echo -e "${YELLOW}🔄 Updating Lambda function codes...${NC}"

# Update data ingestion Lambda
INGESTION_FUNCTION_NAME="poeconomy-data-ingestion"
aws lambda update-function-code \
    --function-name $INGESTION_FUNCTION_NAME \
    --zip-file fileb://$TEMP_DIR/data-ingestion.zip \
    --profile $PROFILE \
    --region $REGION > /dev/null

# Update ML computation Lambda
ML_COMPUTATION_FUNCTION_NAME="poeconomy-ml-computation"
aws lambda update-function-code \
    --function-name $ML_COMPUTATION_FUNCTION_NAME \
    --zip-file fileb://$TEMP_DIR/ml-computation.zip \
    --profile $PROFILE \
    --region $REGION > /dev/null

# Update API serving Lambda
API_SERVING_FUNCTION_NAME="poeconomy-api-serving"
aws lambda update-function-code \
    --function-name $API_SERVING_FUNCTION_NAME \
    --zip-file fileb://$TEMP_DIR/api-serving.zip \
    --profile $PROFILE \
    --region $REGION > /dev/null

echo -e "${GREEN}✅ Lambda functions updated${NC}"

# Set up resource monitoring
echo -e "${YELLOW}📈 Setting up resource monitoring...${NC}"

# Create budget for resource monitoring
aws budgets create-budget \
    --account-id $ACCOUNT_ID \
    --budget '{
        "BudgetName": "PoEconomy-Resource-Monitor",
        "BudgetLimit": {
            "Amount": "50.00",
            "Unit": "USD"
        },
        "TimeUnit": "MONTHLY",
        "BudgetType": "COST",
        "CostFilters": {
            "Service": [
                "Amazon Simple Storage Service",
                "Amazon DynamoDB", 
                "AWS Lambda",
                "Amazon API Gateway",
                "Amazon CloudWatch"
            ]
        }
    }' \
    --profile $PROFILE 2>/dev/null || echo "Budget already exists"

# Create CloudWatch dashboard
aws cloudwatch put-dashboard \
    --dashboard-name "PoEconomy-Usage" \
    --dashboard-body '{
        "widgets": [
            {
                "type": "metric",
                "x": 0,
                "y": 0,
                "width": 12,
                "height": 6,
                "properties": {
                    "metrics": [
                        ["AWS/Lambda", "Invocations", "FunctionName", "'$INGESTION_FUNCTION_NAME'"],
                        [".", ".", ".", "'$ML_COMPUTATION_FUNCTION_NAME'"],
                        [".", ".", ".", "'$API_SERVING_FUNCTION_NAME'"],
                        ["AWS/DynamoDB", "ConsumedReadCapacityUnits", "TableName", "poeconomy-metadata"],
                        [".", "ConsumedWriteCapacityUnits", ".", "."],
                        ["AWS/S3", "NumberOfObjects", "BucketName", "'$BUCKET_NAME'", "StorageType", "AllStorageTypes"]
                    ],
                    "period": 300,
                    "stat": "Sum",
                    "region": "'$REGION'",
                    "title": "PoEconomy Usage Monitoring",
                    "yAxis": {
                        "left": {
                            "min": 0
                        }
                    }
                }
            }
        ]
    }' \
    --profile $PROFILE \
    --region $REGION > /dev/null

echo -e "${GREEN}✅ Resource monitoring configured${NC}"

# Test the deployment
echo -e "${YELLOW}🧪 Testing deployment...${NC}"

# Test API Gateway endpoint
if curl -s --max-time 10 "$API_URL/predict/currencies" > /dev/null; then
    echo -e "${GREEN}✅ API Gateway endpoint responding${NC}"
else
    echo -e "${YELLOW}⚠️  API Gateway endpoint not yet ready (may take a few minutes)${NC}"
fi

# Test Lambda function directly
aws lambda invoke \
    --function-name $API_SERVING_FUNCTION_NAME \
    --payload '{"httpMethod": "GET", "path": "/predict/currencies"}' \
    --profile $PROFILE \
    --region $REGION \
    $TEMP_DIR/test-response.json > /dev/null

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Lambda functions responding${NC}"
else
    echo -e "${YELLOW}⚠️  Lambda function test failed${NC}"
fi

# Cleanup temp directory
rm -rf $TEMP_DIR

# Display deployment summary
echo ""
echo -e "${GREEN}🎉 POECONOMY DEPLOYMENT COMPLETED SUCCESSFULLY! 🎉${NC}"
echo "=================================================="
echo -e "${BLUE}📊 Deployment Summary:${NC}"
echo "• Stack Name: $STACK_NAME"
echo "• Region: $REGION"
echo "• API Gateway URL: $API_URL"
echo "• S3 Bucket: $BUCKET_NAME"
echo "• Architecture: DynamoDB + S3"
echo ""
echo -e "${BLUE}🔗 Quick Links:${NC}"
echo "• AWS Console: https://$REGION.console.aws.amazon.com/cloudformation/home?region=$REGION#/stacks/stackinfo?stackId=$STACK_NAME"
echo "• CloudWatch Dashboard: https://$REGION.console.aws.amazon.com/cloudwatch/home?region=$REGION#dashboards:name=PoEconomy-Usage"
echo "• API Documentation: $API_URL"
echo ""
echo -e "${BLUE}🧪 Test Commands:${NC}"
echo "# Test currency list endpoint"
echo "curl '$API_URL/predict/currencies'"
echo ""
echo "# Test prediction endpoint"
echo "curl -X POST '$API_URL/predict/single' \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"currency\": \"Divine Orb\", \"prediction_horizon_days\": 1}'"
echo ""
echo -e "${BLUE}📈 Monitoring:${NC}"
echo "• Resource Budget: Set at \$50.00/month"
echo "• CloudWatch Dashboard: PoEconomy-Usage"
echo "• Auto-scaling: DynamoDB and Lambda auto-scale based on demand"
echo ""
echo -e "${GREEN}🚀 Architecture Benefits: High performance with auto-scaling capabilities${NC}"
echo -e "${GREEN}🎯 Mission Accomplished: PoEconomy DynamoDB + S3 architecture deployed!${NC}" 