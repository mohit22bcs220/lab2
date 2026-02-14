pipeline {
    agent any

    environment {
        IMAGE_NAME = "scoobydou/bcs220-api"
        CONTAINER_NAME = "bcs220-api-container"
    }

    stages {

        stage('Clone Repository') {
            steps {
                checkout scm
            }
        }

        stage('Train Model') {
            steps {
                sh 'python3 train.py'
            }
        }

        stage('Evaluate Model') {
            steps {
                sh 'python3 evaluate.py'
            }
        }

        stage('Print Metrics with Name & RollNo') {
            steps {
                sh '''
                echo "Model Evaluation Metrics:"
                cat metrics.txt
                echo "Name: Mohit Chaurasia"
                echo "Roll No: 2022BCS0220"
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                sh 'docker build -t $IMAGE_NAME:latest .'
            }
        }

        stage('Push to DockerHub') {
            steps {
                withCredentials([usernamePassword(
                    credentialsId: 'dockerhub-creds',
                    usernameVariable: 'DOCKER_USER',
                    passwordVariable: 'DOCKER_PASS'
                )]) {
                    sh '''
                    echo $DOCKER_PASS | docker login -u $DOCKER_USER --password-stdin
                    docker push $IMAGE_NAME:latest
                    docker logout
                    '''
                }
            }
        }

        stage('Deploy Container') {
            steps {
                sh '''
                docker stop $CONTAINER_NAME || true
                docker rm $CONTAINER_NAME || true
                docker run -d -p 5001:5000 --name $CONTAINER_NAME $IMAGE_NAME:latest
                '''
            }
        }
    }
}
