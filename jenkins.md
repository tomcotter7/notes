# Jenkins

## Groovy

We can use `when` to only execute a stage given a certain condition. For example, we can define a function:
```groovy
    def isPullRequest() {
        return env.CHANGE_ID != null
    }
```

We can then use a `when` clause like so: 
```groovy
    stage('Stage A') { 
        when { 
            expression {
                return isPullRequest()
            }
        }
        steps { ... }
    }
```

This will only execute the steps of the stage if the expression is true.

