# ── Resource group ─────────────────────────────────────────────────────────────

output "resource_group_name" {
  description = "Name of the resource group holding all resources."
  value       = azurerm_resource_group.this.name
}

# ── Cluster ────────────────────────────────────────────────────────────────────

output "cluster_name" {
  description = "AKS cluster name."
  value       = azurerm_kubernetes_cluster.this.name
}

output "kubeconfig_command" {
  description = "Run this command to update your local kubeconfig."
  value       = "az aks get-credentials --resource-group ${azurerm_resource_group.this.name} --name ${azurerm_kubernetes_cluster.this.name}"
}

# ── Container registry ─────────────────────────────────────────────────────────

output "acr_login_server" {
  description = "Azure Container Registry login server (registry URL)."
  value       = azurerm_container_registry.this.login_server
}

output "docker_login_command" {
  description = "Authenticate Docker with ACR."
  value       = "az acr login --name ${azurerm_container_registry.this.name}"
}

# ── Database ───────────────────────────────────────────────────────────────────

output "postgres_fqdn" {
  description = "PostgreSQL Flexible Server fully-qualified domain name."
  value       = azurerm_postgresql_flexible_server.this.fqdn
  sensitive   = true
}

output "postgres_admin_password" {
  description = "Generated PostgreSQL administrator password."
  value       = random_password.postgres.result
  sensitive   = true
}

output "redis_hostname" {
  description = "Azure Cache for Redis hostname."
  value       = azurerm_redis_cache.this.hostname
  sensitive   = true
}

output "redis_primary_access_key" {
  description = "Azure Cache for Redis primary access key."
  value       = azurerm_redis_cache.this.primary_access_key
  sensitive   = true
}

# ── Networking ─────────────────────────────────────────────────────────────────

output "vnet_id" {
  description = "Virtual network ID."
  value       = azurerm_virtual_network.this.id
}
