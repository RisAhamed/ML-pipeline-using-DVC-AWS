# ML-pipeline-using-DVC-AWS
This is Exercise  For implementing Dvc and getting practive with MLOPS operations



# Project Name

## Project Setup and DVC-AWS Integration Guide

### 1. Environment Setup

#### Create and Activate Conda Environment
```bash
# Create new conda environment
conda create -n your_env_name python=3.9

# Activate the environment
conda activate your_env_name
```

#### Install Required Packages
```bash
# Install packages from requirements.txt
pip install -r requirements.txt
```

### 2. AWS Setup

#### 2.1 AWS IAM Setup (One-time setup)
1. Login to AWS Console
2. Go to IAM (Identity and Access Management)
3. Create a new IAM User:
   - Click "Users" → "Add user"
   - Set user name
   - Select "Programmatic access"
4. Set Permissions:
   - Attach existing policies
   - Search and select "AmazonS3FullAccess"
5. Complete user creation and save:
   - Download or copy your credentials:
     - Access key ID
     - Secret access key
   
#### 2.2 AWS CLI Configuration
```bash
# Configure AWS CLI with your credentials
aws configure
```
Enter the following information when prompted:
- AWS Access Key ID
- AWS Secret Access Key
- Default region name (e.g., us-east-1)
- Default output format (json)

#### 2.3 Create S3 Bucket
```bash
# Create a new S3 bucket
aws s3 mb s3://your-bucket-name

# Verify bucket creation
aws s3 ls
```

### 3. DVC Setup and Configuration

#### 3.1 Initialize DVC
```bash
# Initialize DVC in your project
dvc init

# Verify git status
git status
```

#### 3.2 Configure DVC with S3
```bash
# Add your S3 bucket as remote storage
dvc remote add -d storage s3://your-bucket-name/path

# Verify remote storage configuration
dvc remote list
```

#### 3.3 (Optional) Configure DVC with AWS Credentials
If you need to set specific credentials for DVC:
```bash
dvc remote modify storage access_key_id YOUR_ACCESS_KEY_ID
dvc remote modify storage secret_access_key YOUR_SECRET_ACCESS_KEY
dvc remote modify storage region YOUR_AWS_REGION
```

### 4. Using DVC

#### 4.1 Track Data with DVC
```bash
# Add data to DVC
dvc add data/your_data_file

# Commit DVC tracking file
git add data/your_data_file.dvc
git commit -m "Add data tracking"
```

#### 4.2 Push/Pull Data
```bash
# Push data to S3
dvc push

# Pull data from S3
dvc pull
```

### 5. Best Practices

1. **Environment Variables**: Store AWS credentials in environment variables or use `.env` file
2. **GitIgnore**: Ensure your `.gitignore` includes:
   ```
   /data
   .env
   .aws/
   ```
3. **Data Organization**: Keep all data files in a `data/` directory
4. **DVC Files**: Always commit `.dvc` files to Git

### 6. Troubleshooting

1. If `dvc push` fails:
   - Verify AWS credentials
   - Check S3 bucket permissions
   - Ensure bucket name is correct

2. If `dvc pull` fails:
   - Verify internet connection
   - Check if data exists in S3 bucket
   - Verify AWS credentials

### 7. Project Structure
```
project/
│
├── data/               # Data files (DVC-tracked)
├── notebooks/          # Jupyter notebooks
├── src/               # Source code
├── .dvc/              # DVC configuration
├── .gitignore         # Git ignore file
├── requirements.txt   # Project dependencies
└── README.md          # This file
```

### 8. Additional Resources

- [DVC Documentation](https://dvc.org/doc)
- [AWS CLI Documentation](https://aws.amazon.com/cli/)
- [AWS S3 Documentation](https://docs.aws.amazon.com/s3/)

Remember to:
- Replace `your_env_name` with your desired environment name
- Replace `your-bucket-name` with your actual S3 bucket name
- Update the license section according to your project needs
- Modify the project structure section based on your actual project organization
