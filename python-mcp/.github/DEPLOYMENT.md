# Azure Deployment with GitHub Actions

This repository includes GitHub Actions workflows for automated deployment to Azure Web Apps.

## Workflows

### 1. `deploy-azure.yml` - Standard Deployment
- Builds Docker image
- Pushes to Azure Container Registry
- Configures Azure Web App
- Sets startup command and app settings

### 2. `deploy-azure-robust.yml` - Robust Deployment
- Includes retry logic for Azure CLI commands
- Handles connection issues gracefully
- More reliable for production use

## Required GitHub Secrets

To use these workflows, you need to set up the following secrets in your GitHub repository:

### 1. Azure Service Principal Credentials
**Secret Name**: `AZURE_CREDENTIALS`

**Value**: JSON object with service principal details:
```json
{
  "clientId": "your-client-id",
  "clientSecret": "your-client-secret",
  "subscriptionId": "your-subscription-id",
  "tenantId": "your-tenant-id"
}
```

**How to create**:
```bash
# Create service principal
az ad sp create-for-rbac --name "github-actions-papr-mcp" --role contributor --scopes /subscriptions/{subscription-id}/resourceGroups/memory --sdk-auth
```

### 2. Azure Container Registry Password
**Secret Name**: `ACR_PASSWORD`

**Value**: `QAwq2GTLh0aEQP/wtLhKfWAjb3gglV9s`

## Environment Variables

The workflows use these environment variables (can be customized in the workflow file):

- `AZURE_WEBAPP_NAME`: `paprmemorymcp-g0heeseedzcnckbg`
- `ACR_NAME`: `testpaprcontainer`
- `ACR_USERNAME`: `testPaprContainer`
- `DOCKER_IMAGE_NAME`: `papr-mcp-http`

## Workflow Triggers

The workflows are triggered by:
- Push to `main` or `master` branch
- Pull requests to `main` or `master` branch
- Manual trigger via GitHub Actions UI

## Deployment Process

1. **Build**: Creates Docker image with latest code
2. **Test**: Runs pytest tests (optional)
3. **Push**: Pushes image to Azure Container Registry
4. **Configure**: Sets up Azure Web App with:
   - Container image from ACR
   - Registry credentials
   - Startup command: `fastmcp run papr_memory_mcp.core:init_mcp --transport http --host 0.0.0.0 --port 8000`
   - App settings: `PORT=8000`, `WEBSITES_PORT=8000`
5. **Restart**: Restarts the web app
6. **Verify**: Performs health check

## Manual Deployment

If GitHub Actions fails, you can use the PowerShell scripts:

```powershell
# Use the robust script that handles Azure CLI issues
.\scripts\deploy_azure_rest.ps1
```

## Troubleshooting

### Azure CLI Connection Issues
- The robust workflow includes retry logic
- If issues persist, use manual deployment via Azure Portal

### Container Startup Issues
- Verify startup command is set correctly
- Check application settings (PORT, WEBSITES_PORT)
- Review container logs in Azure Portal

### Registry Authentication
- Ensure ACR credentials are correct
- Verify service principal has access to ACR

## Web App URL

After successful deployment, your MCP server will be available at:
**https://paprmemorymcp-g0heeseedzcnckbg.azurewebsites.net**
