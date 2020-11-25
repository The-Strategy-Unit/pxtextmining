library(tidyverse)
library(shinydashboard)
library(reactable)

test_data <- read.csv('C:/Users/andreas.soteriades/Documents/git_projects/positive_about_change_text_mining/y_pred_and_x_test.csv')

body <- dashboardPage(
  dashboardHeader(title = "Patient feedback and its predicted label"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Dashboard", tabName = "dashboard", icon = icon("dashboard")),
      menuItem("Widgets", tabName = "widgets", icon = icon("th"))
    )
  ),
  dashboardBody(
    # Boxes need to be put in a row (or column)
    fluidRow(box(textOutput("modelAccuracyBox"), background = 'red', width = 7)),
    fluidRow(
      #column(7,
             #box(
             #box(textOutput("modelAccuracyBox"), background = 'red'),
               box(width = 7,
                 #box(textOutput("modelAccuracyBox"), background = 'red'),
                 #box(
                   selectInput("pred", "Choose a label:", 
                               choices=sort(unique(test_data$pred))),
                   reactableOutput("pedictedLabels")
                 #)
               ),
             #)
             
      #),
      
      #column(5,
             box(plotOutput("tfidf_bars", width = 5))
      #)
    )
  )
)