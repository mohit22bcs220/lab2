pipeline {
    agent any

    environment {
        IMAGE_NAME = "dockerhub_username/model-api"
        CONTAINER_NAME = "model-api-container"
    }

    stages {

        stage('Clone Repository') {
            steps {
                checkout scm
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    sh 'docker build -t $IMAGE_NAME:latest .'
                }
            }
        }

        stage('Train Model') {
            steps {
                script {
                    sh 'python train.py'
                }
            }
        }

        stage('Evaluate Model') {
            steps {
                script {
                    sh 'python evaluate.py'
                }
            }
        }

        stage('Print Metrics with Name & RollNo') {
            steps {
                script {
                    sh '''
                    echo "Model Evaluation Metrics:"
                    cat metrics.txt
                    echo "Name: YOUR_NAME"
                    echo "Roll No: YOUR_ROLLNO"
                    '''
                }
            }
        }

        stage('Push to DockerHub') {
            steps {
                withCredentials([usernamePassword(credentialsId: 'dockerhub-creds', usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD')]) {
                    sh '''
                    echo $PASSWORD | docker login -u $USERNAME --password-stdin
                    docker push $IMAGE_NAME:latest
                    '''
                }
            }
        }

        stage('Deploy Container') {
            steps {
                script {
                    sh '''
                    docker stop $CONTAINER_NAME || true
                    docker rm $CONTAINER_NAME || true
                    docker run -d -p 5001:5000 --name $CONTAINER_NAME $IMAGE_NAME:latest
                    '''
                }
            }
        }
    }
}
