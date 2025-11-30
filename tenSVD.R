
tenSVD <- function(X, com = NULL, cr = 0) {
  #######################################################################
  # Input:
  #   X:   Input Tensor (multidimensional array)
  #   com: Target Quality (Retained Energy) [0,1]. If NULL, uses 1.
  #   cr:  Minimum Compression Ratio threshold [0,1].
  #
  # Load library(tensr) and library(numbers)
  #
  # Output:
  #   list(TenX, U, Core, Cf, quality, CR)
  #######################################################################
  
  # --- 1. Dimensions Preparation (Reshaping Heuristic) ---
  dimX <- dim(X)
  
  # Calculation of prime factors for all dimensions
  nmode <- sort(unlist(lapply(dimX, primeFactors)))
  
  # Original heuristic logic
  while(length(nmode) >= min(nmode)) {
    new_dim <- nmode[1] * nmode[2]
    nmode <- sort(c(new_dim, nmode[-(1:2)]))
  }
  
  # --- 2. Tensor Reshaping ---
  A1 <- array(as.vector(X), dim = nmode)
  
  # --- 3. HOSVD (Encoding) ---
  m <- dim(A1)
  K <- length(m)
  target_quality <- if(is.null(com)) 1 else com
  
  U <- vector("list", K)
  dm <- 0 
  
  for (k in 1:K) {
    Y_k <- tensr::mat(A1, k)
    Gram <- tcrossprod(Y_k) 
    Eig <- eigen(Gram, symmetric = TRUE)
    U[[k]] <- Eig$vectors
    dm <- dm + length(U[[k]])
  }
  
  # --- 4. Core Tensor Calculation ---
  CoreFull <- tensr::atrans(A1, lapply(U, t))
  
  # --- 5. Thresholding and Compression ---
  S_sq <- CoreFull^2
  sorted_info <- sort.int(as.vector(S_sq), decreasing = TRUE, index.return = TRUE)
  a2 <- sorted_info$x
  indices_linear <- sorted_info$ix
  
  tot_energy <- sum(a2)
  quality_curve <- cumsum(a2) / tot_energy
  
  current_CR_curve <- 1 - (( (seq_along(a2) * 2) + dm ) / prod(m))
  
  stop_idx <- which(quality_curve >= target_quality | current_CR_curve <= cr)[1]
  if(is.na(stop_idx)) stop_idx <- length(a2)
  
  final_quality <- quality_curve[stop_idx]
  final_CR <- current_CR_curve[stop_idx]
  
  # --- 6. Sparse Core Output Construction ---
  top_indices <- indices_linear[1:stop_idx]
  top_values <- CoreFull[top_indices]
  Cf <- cbind(top_indices, top_values)
  
  # --- 7. Reconstruction (Decoding) ---
  CoreRebuilt <- array(0, dim = nmode)
  CoreRebuilt[top_indices] <- top_values
  
  A2 <- CoreRebuilt
  for(i in 1:K) {
    A2 <- tensr::amprod(A2, U[[i]], i)
  }
  
  # --- 8. Final Reshaping and Normalization ---
  XXX <- array(as.vector(A2), dim = dimX)
  
  min_val <- min(XXX)
  max_val <- max(XXX)
  
  if (max_val - min_val > 1e-10) {
    XXX <- (XXX - min_val) / (max_val - min_val)
  }
  
  return(list(
    TenX = XXX, 
    U = U, 
    Core = CoreRebuilt, 
    Cf = Cf,            
    quality = final_quality, 
    CR = final_CR
  ))
}