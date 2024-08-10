import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix

def parse_snp_file_in_chunks(file_path, chunk_size=100000):
    chunks = []
    
    # Read the CSV file in chunks
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # Encode genotypes within each chunk
        chunk['encoded_genotype'] = chunk['genotype'].apply(encode_genotypes)
        chunks.append(chunk)
    
    # Combine all chunks into a single DataFrame
    snp_df = pd.concat(chunks, axis=0)
    return snp_df

def encode_genotypes(genotype):
    if genotype == 'AA':
        return 0
    elif genotype in ['AC', 'GT', 'CT', 'AG']:  # Add any additional heterozygous types if needed
        return 1
    elif genotype == 'CC':
        return 2
    elif genotype in ['TT', '--']:  # Handle missing data
        return 0
    else:
        return np.nan

def main(file_path):
    # Parse the file in chunks
    snp_df = parse_snp_file_in_chunks(file_path)
    
    # Verify the columns
    print("Columns in DataFrame:", snp_df.columns)
    
    # Filter data to a specific chromosome (e.g., chromosome 1) to reduce size
    snp_df = snp_df[snp_df['chromosome'] == 1]
    print(f"Number of rows after filtering by chromosome 1: {len(snp_df)}")

    # Use pivot_table to aggregate data and avoid duplicate entries
    snp_matrix = snp_df.pivot_table(index='rsid', columns='position', values='encoded_genotype', aggfunc='mean')
    
    # Print shape of the pivot table
    print(f"Shape of pivot table: {snp_matrix.shape}")
    
    # Check if the pivot table is empty
    if snp_matrix.empty:
        print("Pivot table is empty. No data to perform PCA.")
        return
    
    # Convert to sparse matrix
    snp_matrix_sparse = csr_matrix(snp_matrix.fillna(0))  # Convert to sparse matrix
    print(f"Shape of sparse matrix: {snp_matrix_sparse.shape}")
    
    # Perform PCA
    pca = PCA(n_components=2)
    try:
        principal_components = pca.fit_transform(snp_matrix_sparse.T)  # Transpose to have features as rows
    except ValueError as e:
        print(f"Error during PCA: {e}")
        return
    
    # Create a DataFrame for plotting
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['SNP'] = snp_matrix.index
    
    # Plotting
    plt.figure(figsize=(10, 7))
    plt.scatter(pca_df['PC1'], pca_df['PC2'])
    for i, snp in enumerate(pca_df['SNP']):
        plt.text(pca_df['PC1'][i], pca_df['PC2'][i], snp, fontsize=9)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of SNP Data (Chromosome 1)')
    plt.grid(True)
    plt.savefig('output.png')  # Save the plot as an image file

# Example usage
file_path = 'sampled_snps.csv'  # Replace with your file path
main(file_path)
