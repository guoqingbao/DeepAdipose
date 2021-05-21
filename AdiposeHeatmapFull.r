# R code to plot heatmap (all 377 radiomics features)
library("openxlsx")
library("stringr")
library(ComplexHeatmap)
library(circlize)
library(colorspace)
library(GetoptLong)
library(ggplot2)
library(gplots)


library("RColorBrewer")

# install.packages('OpasnetUtils')
library(reticulate)

data_path = "./Data/"
output_path = "./Results/"

# load all features
xlfeatures = read.xlsx(paste(data_path , "allBlockData.xlsx",sep = ""), sheet = 1, colNames = TRUE)

xlfeatures = data.frame(xlfeatures)

colnames(xlfeatures)

features = xlfeatures[,3:length(colnames(xlfeatures))]

# fat volumes (ml), age
features$Age <- features$Age*10
features$VFat <- features$VFat*1000
features$SFat <- features$SFat*1000


length(colnames(features))

features[1,1:377]

imagefeatures = features[, 1:(length(colnames(features))-9)]
imagefeatures = scale(imagefeatures)
features[, 1:(length(colnames(features))-9)] = imagefeatures

features = features[c(colnames(imagefeatures), 'Age', 'Gender', 'VFat' ,'SFat', 'IsCTVO', 'IsIR' ,'IsMS')] 

# plot heatmap
ha1 = HeatmapAnnotation(df = features[,378:length(colnames(features))],
    col = list(

               IsCTVO = c("1"="Orange","0"="blue"), IsIR = c("1"="purple","0"="darkgreen"),
                IsMS = c("1"="Brown","0"="green"),
        Gender = c("1"= "pink", "0"= "darkgray")

    ),
                       
     annotation_legend_param = list(
               IsCTVO = list(title = "Visceral Obesity (CT)", at = c("1", "0"), labels = c("Yes", "No")),
               IsIR = list(title = "Insulin Resistance", at = c("1", "0"), labels = c("Yes", "No")),
         VFat = list(title = "Visceral Fat"),
         SFat = list(title = "Subcutaneous Fat"),
      IsMS = list(title = "Metabolic Syndrome", at = c("1", "0"), labels = c("Yes", "No")),
       Gender = list(title = "Gender", at = c("1", "0"), labels = c("Female", "Male"))
         
      )
   )

ht = Heatmap(t(features[,1:377]), name = "Z-Score",row_names_gp = gpar(fontsize = 8),
        show_row_dend = FALSE,show_column_dend = FALSE, cluster_rows = TRUE,
        cluster_columns = TRUE, show_row_names = FALSE, show_column_names = FALSE,clustering_distance_columns = "pearson",
        show_heatmap_legend = TRUE,
       ,top_annotation=ha1)



p = draw(ht, heatmap_legend_side = "right")
decorate_annotation("IsIR", {grid.text("Insulin Resistance", unit(1, "npc") + unit(2, "mm"), 0.5, default.units = "npc", just = "left" , gp = gpar(fontsize = 10))})
decorate_annotation("IsCTVO", {grid.text("Visceral Obesity (CT)", unit(1, "npc") + unit(2, "mm"), 0.5, default.units = "npc", just = "left" , gp = gpar(fontsize = 10))})
decorate_annotation("IsMS", {grid.text("Metabolic Syndrome", unit(1, "npc") + unit(2, "mm"), 0.5, default.units = "npc", just = "left" , gp = gpar(fontsize = 10))})
decorate_annotation("Age", {grid.text("Age", unit(1, "npc") + unit(2, "mm"), 0.5, default.units = "npc", just = "left" , gp = gpar(fontsize = 10))})
decorate_annotation("Gender", {grid.text("Gender", unit(1, "npc") + unit(2, "mm"), 0.5, default.units = "npc", just = "left" , gp = gpar(fontsize = 10))})
decorate_annotation("SFat", {grid.text("Subcutaneous Fat(ml)", unit(1, "npc") + unit(2, "mm"), 0.5, default.units = "npc", just = "left" , gp = gpar(fontsize = 10))})
decorate_annotation("VFat", {grid.text("Viceral Fat(ml)", unit(1, "npc") + unit(2, "mm"), 0.5, default.units = "npc", just = "left" , gp = gpar(fontsize = 10))})

# save to file
title = paste(output_path , "inner_block_for_heatmap_allfeatures.tif",sep = "")
tiff(title, width=900*3, height=800*3, units="px", res=96*3, compression = "lzw")
p = draw(ht, heatmap_legend_side = "right")
decorate_annotation("IsIR", {grid.text("Insulin Resistance", unit(1, "npc") + unit(2, "mm"), 0.5, default.units = "npc", just = "left" , gp = gpar(fontsize = 10))})
decorate_annotation("IsCTVO", {grid.text("Visceral Obesity (CT)", unit(1, "npc") + unit(2, "mm"), 0.5, default.units = "npc", just = "left" , gp = gpar(fontsize = 10))})
decorate_annotation("IsMS", {grid.text("Metabolic Syndrome", unit(1, "npc") + unit(2, "mm"), 0.5, default.units = "npc", just = "left" , gp = gpar(fontsize = 10))})
decorate_annotation("Age", {grid.text("Age", unit(1, "npc") + unit(2, "mm"), 0.5, default.units = "npc", just = "left" , gp = gpar(fontsize = 10))})
decorate_annotation("Gender", {grid.text("Gender", unit(1, "npc") + unit(2, "mm"), 0.5, default.units = "npc", just = "left" , gp = gpar(fontsize = 10))})
decorate_annotation("SFat", {grid.text("Subcutaneous Fat(ml)", unit(1, "npc") + unit(2, "mm"), 0.5, default.units = "npc", just = "left" , gp = gpar(fontsize = 10))})
decorate_annotation("VFat", {grid.text("Viceral Fat(ml)", unit(1, "npc") + unit(2, "mm"), 0.5, default.units = "npc", just = "left" , gp = gpar(fontsize = 10))})
dev.off()






