#!/usr/bin/env Rscript

## Load packages
library(ggplot2)
library(plyr)

## Read the data set
frame = read.csv("table.csv")

## Load labels and labelling functions
source("labels.r")

## Define a function to convert nanoseconds to seconds
nsec2sec = function(nsec)
{
  return(nsec / 1000000000)
}

gradient = function(colors = c("#0000FF", "#11FFFF", "#22FF22", "#FFFF33", "#FF4444", "#FF55FF"), grayscale = c("#FDFDFD", "#020202"), number = 10)
{
	## Steps is a vector of rgb colors the generated gradient must use
	## number is the number of colors to be generated

	segment = function(x, y, i, begin)
	{
		if(begin)
		{
			return(floor((i - 1) / (x - 1) * (y - 1) ) + 1)
		}
		else
		{
			return(ceiling((i - 1) / (x - 1) * (y - 1) ) + 1)
		}
	}

	ratio = function(x, y, i)
	{
		return((i - 1) / (x - 1) * (y - 1) + 1 - segment(x, y, i, TRUE))
	}


	x = number
	y = length(colors)

	yy = length(grayscale)

	out = list()

	for(i in 1:number)
	{
		r = ratio(x, y, i)
		begin = segment(x, y, i, TRUE)
		end = segment(x, y, i, FALSE)
		#cat(paste(sep = "", "i: ", as.character(i), "; x: ", as.character(x), "; y = ", as.character(y), "; begin: ", as.character(begin), "; end: ", as.character(end), "; r: ", as.character(r)))

		rbegin = col2rgb(colors[begin], alpha = TRUE)["red",]
		gbegin = col2rgb(colors[begin], alpha = TRUE)["green",]
		bbegin = col2rgb(colors[begin], alpha = TRUE)["blue",]
		abegin = col2rgb(colors[begin], alpha = TRUE)["alpha",]

		rend = col2rgb(colors[end], alpha = TRUE)["red",]
		gend = col2rgb(colors[end], alpha = TRUE)["green",]
		bend = col2rgb(colors[end], alpha = TRUE)["blue",]
		aend = col2rgb(colors[end], alpha = TRUE)["alpha",]

		#cat(paste(sep="", "rbegin: ", as.character(rbegin), "; gbegin: ", as.character(gbegin), "; bbegin: ", as.character(bbegin), "; abegin: ", as.character(abegin), ".\n"))
		#cat(paste(sep="", "rend: ", as.character(rend), "; gend: ", as.character(gend), "; bend: ", as.character(bend), "; aend: ", as.character(aend), ".\n"))

		## Create the intermediate color
		rout = rbegin * (1 - r) + rend * r
		gout = gbegin * (1 - r) + gend * r
		bout = bbegin * (1 - r) + bend * r
		aout = abegin * (1 - r) + aend * r

		#cat(paste(sep="", "rout: ", as.character(rout), "; gout: ", as.character(gout), "; bout: ", as.character(bout), "; aout: ", as.character(aout), ".\n"))

		rr = ratio(x, yy, i)
		gbegin = segment(x, yy, i, TRUE)
		gend = segment(x, yy, i, FALSE)
		#cat(paste(sep = "", "i: ", as.character(i), "; x: ", as.character(x), "; yy = ", as.character(yy), "; gbegin: ", as.character(gbegin), "; gend: ", as.character(gend), "; rr: ", as.character(rr)))

		## Integrate luminosity into the gradient, for grayscale outputs
		hsvbegin = rgb2hsv(r = col2rgb(grayscale[gbegin], alpha = TRUE)["red",], g = col2rgb(grayscale[gbegin], alpha = TRUE)["green",], b = col2rgb(grayscale[gbegin], alpha = TRUE)["blue",], maxColorValue = 255)
		hsvend = rgb2hsv(r = col2rgb(grayscale[gend], alpha = TRUE)["red",], g = col2rgb(grayscale[gend], alpha = TRUE)["green",], b = col2rgb(grayscale[gend], alpha = TRUE)["blue",], maxColorValue = 255)
		hsvout = rgb2hsv(r = rout, g = gout, b = bout, maxColorValue = 255)

		hsvout[3] = hsvbegin[3] * (1 - rr) + hsvend[3] * rr

		out = append(out, hsv(hsvout[1], hsvout[2], hsvout[3], aout / 255))
		#cat("####################\n\n")
	}

	return(unlist(out))
}

## Compute speedup for each run
frame = ddply(
	frame,
	c("size", "pattern", "nb_threads", "implementation", "input_instance", "run_instance", "value"),
	summarize,
	time = time,
	speedup = frame[frame$size == size & frame$pattern == pattern & frame$nb_threads == "seq" & frame$implementation == implementation & frame$input_instance == input_instance & frame$run_instance == run_instance & frame$value == value, "time"] / time
)

## Compute mean and standard deviation
frame = ddply(
	frame,
	c("size", "pattern", "nb_threads", "implementation"),
	summarize,
	mean_time = mean(time),
	std_time = sd(time),
	mean_speedup = mean(speedup),
	std_speedup = sd(speedup)
)

## Create a simple plot with ggplots
plot = ggplot() +
	geom_line(data = apply_labels(frame[as.integer(frame$nb_threads) == max(as.integer(frame$nb_threads)),]),
		aes(size, mean_time, group = interaction(pattern, implementation), color = pattern),
		size = 1) +
	geom_point(data = apply_labels(frame[as.integer(frame$nb_threads) == max(as.integer(frame$nb_threads)),]),
		aes(size, mean_time, group = interaction(pattern, implementation), color = pattern, shape = implementation),
		#aes(size, mean_time, group = interaction(pattern, implementation), color = interaction(pattern, implementation)),
		size = 5) +
	guides(fill = guide_legend(title = "Time dependency on input size"), color = guide_legend(title = "Pattern"), shape = guide_legend(title = "Implementation")) +
	ylab("Running time in milliseconds") +
	xlab(label("size")) +
	ggtitle(paste(sep = "", "Time to sort using ", as.character(max(as.integer(frame$nb_threads))), " cores." )) 

### Save the plot as a svg file
ggsave(file = "time_size.svg", plot = plot, width = 8, height = 6)

## Create a simple plot with ggplots
plot = ggplot() +
	geom_line(data = apply_labels(frame[frame$size == max(frame$size),]),
		aes(nb_threads, mean_time, group = interaction(pattern, implementation), color = pattern),
		size = 1) +
	geom_point(data = apply_labels(frame[frame$size == max(frame$size),]),
		aes(nb_threads, mean_time, group = interaction(pattern, implementation), color = pattern, shape = implementation),
		size = 5) +
	geom_errorbar(data = apply_labels(frame[frame$size == max(frame$size),]),
		aes(nb_threads, ymax = mean_time + std_time, ymin = mean_time - std_time, group = interaction(pattern, implementation), color = pattern, pattern = implementation, width = 0.25)) +
	guides(fill = guide_legend(title = "Time function of number of threads"), color = guide_legend(title = "Input pattern"), shape = guide_legend(title = "Implementation")) +
	ylab("Running time in milliseconds") +
	xlab(label("nb_threads")) +
	ggtitle(paste(sep = "", "Time to sort ", as.character(max(frame$size)), " integers." )) 

## Save the plot as a svg file
ggsave(file = "time_threads.svg", plot = plot, width = 8, height = 6)

## Create a simple plot with ggplots
plot = ggplot() +
	geom_line(data = apply_labels(frame[frame$size == max(frame$size),]),
		aes(nb_threads, mean_speedup, group = interaction(pattern, implementation), color = pattern),
		size = 1) +
	geom_point(data = apply_labels(frame[frame$size == max(frame$size),]),
		aes(nb_threads, mean_speedup, group = interaction(pattern, implementation), color = pattern, shape = implementation),
		size = 5) +
	geom_errorbar(data = apply_labels(frame[frame$size == max(frame$size),]),
		aes(nb_threads, ymax = mean_speedup + std_speedup, ymin = mean_speedup - std_speedup, group = interaction(pattern, implementation), color = pattern, pattern = implementation, width = 0.25)) +
	guides(fill = guide_legend(title = "Speedup function of number of threads"), color = guide_legend(title = "Input pattern"), shape = guide_legend(title = "Implementation")) +
	ylab("Speedup") +
	xlab(label("nb_threads")) +
	ggtitle(paste(sep = "", "Time to sort ", as.character(max(frame$size)), " integers." )) 

## Save the plot as a svg file
ggsave(file = "speedup_threads.svg", plot = plot, width = 8, height = 6)

## Numbers necessary for survey
show("Please report this table to the survey sheet")
options(digits=3)
matrix = frame[frame$pattern == "uniform-random" & frame$size == max(frame$size),c("implementation", "nb_threads", "mean_speedup")]
matrix = apply_labels(matrix)
show(matrix[,c("implementation", "nb_threads", "mean_speedup")])
