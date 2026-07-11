# ── Global ─────────────────────────────────────────────────────────────────────

variable "location" {
  description = "Azure region for all resources (e.g. eastus, westeurope)."
  type        = string
  default     = "eastus"
}

variable "environment" {
  description = "Deployment environment (staging | production)."
  type        = string
  default     = "staging"
  validation {
    condition     = contains(["staging", "production"], var.environment)
    error_message = "environment must be 'staging' or 'production'."
  }
}

variable "project" {
  description = "Short project name — used as a prefix for all resource names."
  type        = string
  default     = "healthcare"
}

# ── Networking ─────────────────────────────────────────────────────────────────

variable "vnet_cidr" {
  description = "Address space for the virtual network."
  type        = string
  default     = "10.0.0.0/16"
}

variable "aks_subnet_cidr" {
  description = "Subnet CIDR for the AKS node pool."
  type        = string
  default     = "10.0.1.0/24"
}

variable "data_subnet_cidr" {
  description = "Subnet CIDR for the managed data services (Postgres, Redis)."
  type        = string
  default     = "10.0.2.0/24"
}

# ── AKS ────────────────────────────────────────────────────────────────────────

variable "kubernetes_version" {
  description = "Kubernetes version for the AKS cluster."
  type        = string
  default     = "1.29"
}

variable "node_vm_size" {
  description = "VM size for the AKS default node pool."
  type        = string
  default     = "Standard_D2s_v5"
}

variable "node_min_count" {
  description = "Minimum number of AKS nodes (cluster autoscaler)."
  type        = number
  default     = 2
}

variable "node_max_count" {
  description = "Maximum number of AKS nodes (cluster autoscaler)."
  type        = number
  default     = 6
}

# ── Azure Database for PostgreSQL ──────────────────────────────────────────────

variable "postgres_sku_name" {
  description = "SKU name for the PostgreSQL Flexible Server (e.g. B_Standard_B1ms, GP_Standard_D2s_v3)."
  type        = string
  default     = "B_Standard_B1ms"
}

variable "postgres_storage_mb" {
  description = "Storage for the PostgreSQL Flexible Server in MB."
  type        = number
  default     = 32768
}

variable "postgres_version" {
  description = "PostgreSQL major version."
  type        = string
  default     = "15"
}

variable "postgres_database_name" {
  description = "Name of the initial database."
  type        = string
  default     = "healthcare"
}

variable "postgres_admin_username" {
  description = "Administrator username for the PostgreSQL server."
  type        = string
  default     = "healthcare_app"
}

# ── Azure Cache for Redis ──────────────────────────────────────────────────────

variable "redis_capacity" {
  description = "Redis cache size (0-6 for Basic/Standard, 1-5 for Premium)."
  type        = number
  default     = 0
}

variable "redis_family" {
  description = "Redis SKU family: C (Basic/Standard) or P (Premium)."
  type        = string
  default     = "C"
}

variable "redis_sku_name" {
  description = "Redis SKU: Basic, Standard, or Premium."
  type        = string
  default     = "Basic"
}

# ── Azure Container Registry ───────────────────────────────────────────────────

variable "acr_sku" {
  description = "Container registry SKU: Basic, Standard, or Premium."
  type        = string
  default     = "Basic"
}
