library(tuneR)
library(seewave)

#reading the wae file and taking FFT on a 500ms window
s <- tuneR::readWave('C:\\Users\\Chaitanya\\Documents\\Gender\\male and female\\female1.wav', from = 0.5, to = 1, units = "seconds")
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

features <- list(c('meanfun', 'Q25', 'sd', 'IQR', 'sfm', 'meanfreq', 'mod'),
                 c(meanfun, Q25, sd, IQR, sfm, meanfreq, mod))

df <- data.frame(features)
write.csv(df, file="female1.csv", col.names = 0)
