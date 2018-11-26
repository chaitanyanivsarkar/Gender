library(tuneR)
library(seewave)

fetrs <- function(f, d, a=0, b=1) {
        #reading the wae file and taking FFT on a 50ms window
        s <- tuneR::readWave(filename = f, from = a, to = b, units = "seconds")
        X <- seewave::spec(s, f = s@samp.rate, plot = FALSE)
        
        #making a list of all frequencies of a signal
        Xfe <- seewave::specprop(X, f = s@samp.rate, flim = c(0, 280/1000), plot = FALSE)
        fundamenal <- seewave::fund(s, f = s@samp.rate, ovlp = 50, threshold = 5, fmax = 280, ylim=c(0, 280/1000), plot = FALSE, wl = 2048)[, 2]
        
        meanfreq <- Xfe$mean/1000
        sd <- Xfe$sd/1000
        Q25 <- Xfe$Q25/1000 
        IQR <- Xfe$IQR/1000
        sfm <- Xfe$sfm
        mod <- Xfe$mode/1000
        meanfun <- mean(fundamenal, na.rm=TRUE)
        
        features <- c(meanfun, Q25, sd, IQR, sfm, meanfreq, mod)
        
        df <- data.frame(features)
        write.csv(df, file=d)
        return(features)
}

# feature extraction on the Demonstration set
feat <- fetrs("C:\\Users\\Chaitanya\\Documents\\Gender\\male and female\\female1.wav" ,"female1.csv" ,0 ,1)
feat <- fetrs("C:\\Users\\Chaitanya\\Documents\\Gender\\male and female\\female2.wav" ,"female2.csv" ,0 ,1)
feat <- fetrs("C:\\Users\\Chaitanya\\Documents\\Gender\\male and female\\female3.wav" ,"female3.csv" ,0 ,1)
feat <- fetrs("C:\\Users\\Chaitanya\\Documents\\Gender\\male and female\\male1.wav" ,"male1.csv" ,0.5 ,1)
feat <- fetrs("C:\\Users\\Chaitanya\\Documents\\Gender\\male and female\\male2.wav" ,"male2.csv" ,0.5 ,1)
feat <- fetrs("C:\\Users\\Chaitanya\\Documents\\Gender\\male and female\\male3.wav" ,"male3.csv" ,0.5 ,1)
