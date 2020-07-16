package io.github.jhipster.sample.prediction.entropy;

public class CalculateShannonEntropy {
    public double entropy() {
        var entropy = 0.0;
        var m = 1;
        var n = 9;
        int[] freq = new int[m+1];
        for(int i=1;i<=m;i++) {
            var p = 1.0*freq[i]/n;
                if(freq[i] > 0) {
                    entropy -= p*Math.log(p)/Math.log(2);
                }
        }
        return entropy;
    }
}
