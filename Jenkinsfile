pipeline {
    agent any

    environment {
            IMAGE_NAME = "scoobydou/bcs220-api"
            CONTAINER_NAME = "bcs220-api-container"
        }

    stages {

        stage('Checkout Code') {
            steps {
                checkout scm
            }
        }

        stage('Install Dependencies') {
            steps {
                sh '''
                python3 -m venv venv
                . venv/bin/activate
                pip install --upgrade pip
                pip install -r requirements.txt
                '''
            }
        }

        stage('Train Model') {
            steps {
                sh '''
                . venv/bin/activate
                python3 train.py
                '''
            }
        }

        stage('Evaluate Model') {
            steps {
                sh '''
                . venv/bin/activate
                python3 evaluate.py || echo "No evaluate.py found"
                '''
            }
        }

        stage('Print Metrics with Name & RollNo') {
            steps {
                sh '''
                echo "===== Model Metrics ====="
                cat outputs/results/experiments.json
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
                    usernameVariable: 'USERNAME',
                    passwordVariable: 'PASSWORD'
                )]) {
                    sh '''
                    echo $PASSWORD | docker login -u $USERNAME --password-stdin
                    docker push $IMAGE_NAME:latest
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
