# Event-Driven Autoscaling for Kubernetes Workloads with KEDA and Prometheus

This repository demonstrates how to implement sophisticated autoscaling in Kubernetes using KEDA (Kubernetes Event-Driven Autoscaler) with Prometheus metrics. The setup showcases horizontal pod autoscaling based on custom application metrics and automatic node provisioning with Karpenter.

## Architecture

The solution combines several key components to create a fully automated, event-driven scaling system:

- **KEDA**: Provides event-driven autoscaling capabilities, translating custom metrics into HPA-compatible scaling decisions
- **Prometheus**: Collects and stores application metrics used for scaling decisions
- **Karpenter**: Automatically provisions and manages EC2 instances based on pod scheduling requirements
- **Flux**: GitOps continuous delivery for declarative cluster management
- **Grafana**: Observability dashboard for monitoring scaling behavior and metrics

![architecture diagram](./images/architecture_diagram.png)

## Prerequisites

- AWS account with appropriate permissions
- **Fork this repository** (Flux will commit its configuration files to your fork)
- Linux-based operating system for command execution

## Deployment

### 1. Environment Setup

First, **fork this repository**, then configure the required environment variables:

```bash
export CLUSTER_NAME={your cluster name}
export AWS_DEFAULT_REGION={your region}
export GITHUB_TOKEN={GitHub token}
export GITHUB_USER={GitHub user}
export GITHUB_REPO={your forked repo}
```

### 2. Create Amazon EKS Cluster

Deploy an EKS cluster with OIDC and auto mode enabled:

```bash
cat << EOF | eksctl create cluster -f -
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: ${CLUSTER_NAME}
  region: ${AWS_DEFAULT_REGION}

iam:
  withOIDC: true

autoModeConfig:
  enabled: true

EOF
```

### 3. Install and Bootstrap Flux

Install Flux CLI and bootstrap GitOps:

```bash
brew install fluxcd/tap/flux
flux bootstrap github \
--owner=${GITHUB_USER} \
--repository=${GITHUB_REPO} \
--branch=main \
--personal \
--path=eda-keda-prometheus/cluster
```

### 4. Monitor Deployment

Watch the deployment progress:

```bash
flux get kustomizations --watch
```

### 5. Verify Initial Setup

Check that nodes are provisioned correctly:

```bash
kubectl get nodes -o json|jq -Cjr '.items[] | .metadata.name," ",.metadata.labels."beta.kubernetes.io/instance-type"," ",.metadata.labels."beta.kubernetes.io/arch", "\n"'|sort -k3 -r
```

Expected output:
```bash
i-09f8cc029971ea2bd c6g.large arm64
i-0cdc488091ff514bd c5.large amd64
i-0ca4843df75cdea84 c5a.large amd64
```

## Understanding Kubernetes API Aggregation and KEDA Integration

The Kubernetes API aggregation layer extends the cluster's API surface without modifying core Kubernetes code. This architecture enables three critical metric APIs:

- **`metrics.k8s.io`**: Basic pod and node resource metrics (CPU, memory)
- **`custom.metrics.k8s.io`**: Application-specific metrics from within the cluster
- **`external.metrics.k8s.io`**: Metrics from external systems (like Prometheus)

KEDA acts as an adapter, implementing both custom and external metrics APIs. It translates various event sources into metrics that the Horizontal Pod Autoscaler (HPA) can consume for scaling decisions.

### Examining KEDA Metrics Exposure

Check the external metrics API that KEDA exposes:

```bash
kubectl get --raw "/apis/external.metrics.k8s.io/v1beta1/namespaces/workload/s0-prometheus?labelSelector=scaledobject.keda.sh%2Fname%3Dexample-app" | jq
```

Expected output:
```bash
{
  "kind": "ExternalMetricValueList",
  "apiVersion": "external.metrics.k8s.io/v1beta1",
  "metadata": {},
  "items": [
    {
      "metricName": "s0-prometheus",
      "metricLabels": null,
      "timestamp": "2025-09-14T07:05:26Z",
      "value": "0"
    }
  ]
}
```

The `s0-prometheus` naming convention serves multiple purposes:
- **`s0-`**: Prefix indicating "ScaledObject" to ensure unique metric names
- **`prometheus`**: Identifies the trigger type/source
- **Uniqueness**: Prevents naming conflicts across different ScaledObjects

### HPA Integration Analysis

In this example, we'll use the `http_requests_total` metric to HPA functionality. This metric counts total HTTP requests and provides an easy way to trigger scaling events by simply generating traffic to our service.

While this works well for learning and testing purposes, it's important to note that in production environments, you'll need to carefully select metrics that truly reflect your application's workload and performance requirements. Production metrics will vary significantly based on your specific use case. For instance, a message processing service might scale based on queue length, while a customer-facing API might scale on request latency. Compute-intensive applications often scale on CPU utilization, and connection-bound services might scale based on concurrent connections. Data processing applications commonly use memory consumption as their scaling metric.

The key is selecting metrics that align with your application's bottlenecks and business requirements. For now, http_requests_total serves as a clear, easily controllable metric for understanding how HPA works.

Examine the HPA that KEDA automatically creates:

```bash
kubectl get hpa -A
```

Expected output:
```bash
NAMESPACE   NAME                   REFERENCE                           TARGETS     MINPODS   MAXPODS   REPLICAS   AGE
workload    keda-hpa-example-app   Deployment/prometheus-example-app   0/2 (avg)   1         10        1          2m58s
```

Get detailed HPA information:

```bash
kubectl describe hpa keda-hpa-example-app -n workload
```

Expected output:
```bash
Name:                                      keda-hpa-example-app
Namespace:                                 workload
Labels:                                    app.kubernetes.io/managed-by=keda-operator
                                           app.kubernetes.io/name=keda-hpa-example-app
                                           app.kubernetes.io/part-of=example-app
                                           app.kubernetes.io/version=2.17.2
                                           kustomize.toolkit.fluxcd.io/name=workload
                                           kustomize.toolkit.fluxcd.io/namespace=flux-system
                                           scaledobject.keda.sh/name=example-app
Annotations:                               <none>
CreationTimestamp:                         Sun, 14 Sep 2025 09:04:08 +0200
Reference:                                 Deployment/prometheus-example-app
Metrics:                                   ( current / target )
  "s0-prometheus" (target average value):  0 / 2
Min replicas:                              1
Max replicas:                              10
Deployment pods:                           1 current / 1 desired
Conditions:
  Type            Status  Reason               Message
  ----            ------  ------               -------
  AbleToScale     True    ScaleDownStabilized  recent recommendations were higher than current one, applying the highest recent recommendation
  ScalingActive   True    ValidMetricFound     the HPA was able to successfully calculate a replica count from external metric s0-prometheus(&LabelSelector{MatchLabels:map[string]string{scaledobject.keda.sh/name: example-app,},MatchExpressions:[]LabelSelectorRequirement{},})
  ScalingLimited  False   DesiredWithinRange   the desired count is within the acceptable range
Events:           <none>
```

The key metric configuration shows:
```bash
Metrics:                                   ( current / target )
  "s0-prometheus" (target average value):  0 / 2
```

This corresponds to the ScaledObject configuration in `eda-keda-prometheus/workload/scaledobject.yaml`:
```bash
      metricName: http_requests_total
      query: sum(http_requests_total)
      threshold: "2"
```

### HPA Configuration Deep Dive

Examine the complete HPA YAML configuration:

```bash
kubectl get hpa keda-hpa-example-app -n workload -o yaml
```

Expected output:

```bash
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  creationTimestamp: "2025-09-14T07:04:08Z"
  labels:
    app.kubernetes.io/managed-by: keda-operator
    app.kubernetes.io/name: keda-hpa-example-app
    app.kubernetes.io/part-of: example-app
    app.kubernetes.io/version: 2.17.2
    kustomize.toolkit.fluxcd.io/name: workload
    kustomize.toolkit.fluxcd.io/namespace: flux-system
    scaledobject.keda.sh/name: example-app
  name: keda-hpa-example-app
  namespace: workload
  ownerReferences:
  - apiVersion: keda.sh/v1alpha1
    blockOwnerDeletion: true
    controller: true
    kind: ScaledObject
    name: example-app
    uid: a24813ff-033b-4d15-bcca-a9e3403064bd
  resourceVersion: "351305"
  uid: dd09e5a8-36fd-42f0-96b5-ed133ac949fb
spec:
  maxReplicas: 10
  metrics:
  - external:
      metric:
        name: s0-prometheus
        selector:
          matchLabels:
            scaledobject.keda.sh/name: example-app
      target:
        averageValue: "2"
        type: AverageValue
    type: External
  minReplicas: 1
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: prometheus-example-app
status:
  conditions:
  - lastTransitionTime: "2025-09-14T07:04:23Z"
    message: recommended size matches current size
    reason: ReadyForNewScale
    status: "True"
    type: AbleToScale
  - lastTransitionTime: "2025-09-14T07:04:23Z"
    message: 'the HPA was able to successfully calculate a replica count from external
      metric s0-prometheus(&LabelSelector{MatchLabels:map[string]string{scaledobject.keda.sh/name:
      example-app,},MatchExpressions:[]LabelSelectorRequirement{},})'
    reason: ValidMetricFound
    status: "True"
    type: ScalingActive
  - lastTransitionTime: "2025-09-14T07:09:24Z"
    message: the desired replica count is less than the minimum replica count
    reason: TooFewReplicas
    status: "True"
    type: ScalingLimited
  currentMetrics:
  - external:
      current:
        averageValue: "0"
      metric:
        name: s0-prometheus
        selector:
          matchLabels:
            scaledobject.keda.sh/name: example-app
    type: External
  currentReplicas: 1
  desiredReplicas: 1
```

Critical configuration details:
- **`type: AverageValue`**: The metric value is divided by the number of pods before comparison with the target
- **`type: External`**: Indicates the metric comes from an external source (Prometheus via KEDA)
- **`averageValue: "2"`**: Target threshold for scaling decisions

## Testing Autoscaling Behavior

### 1. Setup Port Forwarding

Forward traffic to the example application:

```bash
kubectl port-forward svc/prometheus-example-app 3001:8080 -n workload &
```

### 2. Generate Initial Traffic

Send a single request to establish baseline metrics:

```bash
curl http://localhost:3001/
```

Check the updated metric value:

```bash
kubectl get --raw "/apis/external.metrics.k8s.io/v1beta1/namespaces/workload/s0-prometheus?labelSelector=scaledobject.keda.sh%2Fname%3Dexample-app" | jq
```

Expected output:
```bash
{
  "kind": "ExternalMetricValueList",
  "apiVersion": "external.metrics.k8s.io/v1beta1",
  "metadata": {},
  "items": [
    {
      "metricName": "s0-prometheus",
      "metricLabels": null,
      "timestamp": "2025-09-14T07:45:48Z",
      "value": "1"
    }
  ]
}
```

### 3. Trigger Scaling Event

Generate sufficient traffic to exceed the scaling threshold:

```bash
for i in {1..30}
do
    curl http://localhost:3001/
done
```

Verify the metric increase:

```bash
kubectl get --raw "/apis/external.metrics.k8s.io/v1beta1/namespaces/workload/s0-prometheus?labelSelector=scaledobject.keda.sh%2Fname%3Dexample-app" | jq
```

Expected output:
```bash
{
  "kind": "ExternalMetricValueList",
  "apiVersion": "external.metrics.k8s.io/v1beta1",
  "metadata": {},
  "items": [
    {
      "metricName": "s0-prometheus",
      "metricLabels": null,
      "timestamp": "2025-09-14T13:10:16Z",
      "value": "31"
    }
  ]
}
```

### 4. Observe Pod Scaling

Monitor pod creation as the HPA responds to increased metrics:

```bash
kubectl get pods -n workload
```

Expected output showing scaling in progress:
```bash
NAME                                     READY   STATUS              RESTARTS   AGE
prometheus-example-app-69986d5b8-2s965   0/1     ContainerCreating   0          22s
prometheus-example-app-69986d5b8-46pbh   0/1     ContainerCreating   0          22s
prometheus-example-app-69986d5b8-5kmqp   0/1     Pending             0          7s
prometheus-example-app-69986d5b8-62prv   0/1     ContainerCreating   0          22s
prometheus-example-app-69986d5b8-94jhs   1/1     Running             0          176m
prometheus-example-app-69986d5b8-9cwg8   0/1     Pending             0          7s
prometheus-example-app-69986d5b8-ct85v   0/1     Pending             0          7s
prometheus-example-app-69986d5b8-xnhlh   0/1     Pending             0          7s
```

### 5. Verify Node Autoscaling

Check that Karpenter has provisioned additional nodes to accommodate the scaled pods:

```bash
kubectl get nodes -o json|jq -Cjr '.items[] | .metadata.name," ",.metadata.labels."beta.kubernetes.io/instance-type"," ",.metadata.labels."beta.kubernetes.io/arch", "\n"'|sort -k3 -r
```

Expected output showing new nodes:
```bash
i-09f8cc029971ea2bd c6g.large arm64
i-0cdc488091ff514bd c5.large amd64
i-0ca4843df75cdea84 c5a.large amd64
i-06f90868d790a5908 c5.large amd64
i-05f57c38b7177eb1d c5.xlarge amd64
```

Examine Karpenter NodeClaims:

```bash
kubectl get nodeclaim -A
```

Expected output showing the recently created claims:
```bash
NAME                    TYPE        CAPACITY    ZONE              NODE                  READY   AGE
demo-workload-5ft6t     c5.large    on-demand   ap-northeast-1d   i-06f90868d790a5908   True    87s
demo-workload-7x6m4     c5.xlarge   on-demand   ap-northeast-1c   i-05f57c38b7177eb1d   True    72s
demo-workload-d8q4c     c5.large    on-demand   ap-northeast-1c   i-0cdc488091ff514bd   True    12m
general-purpose-lhmcz   c5a.large   on-demand   ap-northeast-1c   i-0ca4843df75cdea84   True    4h17m
system-prjh2            c6g.large   on-demand   ap-northeast-1a   i-09f8cc029971ea2bd   True    4h22m
```

## Observability

### Grafana Dashboard Access

Retrieve the Grafana admin password and setup port forwarding:

```bash
kubectl get secret kube-prometheus-stack-grafana -n monitoring -o jsonpath="{.data.admin-password}" | base64 --decode ; echo

kubectl port-forward svc/kube-prometheus-stack-grafana 3000:80 -n monitoring &
```

Access the Grafana UI and import an HPA dashboard, e.g.: https://grafana.com/grafana/dashboards/22128-horizontal-pod-autoscaler-hpa/

You will see the HPA event we just triggered:

![grafana dashboard hpa](./images/grafana_dashboard_hpa.png)

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.