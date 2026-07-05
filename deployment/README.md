# Local Kubernetes Deployment

```bash
# Build the image inside Minikube's Docker daemon.
eval $(minikube docker-env)
docker build -t heart-disease-api:1.0.0 .

kubectl apply -f deployment/configmap.yaml
kubectl apply -f deployment/deployment.yaml
kubectl apply -f deployment/service.yaml

kubectl get pods
kubectl get services
minikube service heart-disease-api-service --url
```

Test the URL returned by Minikube:

```bash
curl -X POST "<MINIKUBE_URL>/predict" \
  -H "Content-Type: application/json" \
  --data @sample_request.json
```
