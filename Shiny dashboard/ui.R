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
    fluidRow(
      box(
        textOutput("modelAccuracyBox"), background = 'red', width = 7
      )
    ),
    
    fluidRow(
      box(width = 7,
          selectInput("pred", "Choose a label:", 
                      choices=sort(unique(test_data$pred))),
          reactableOutput("pedictedLabels")
      ),
      
      fluidRow(
        box(plotOutput("tfidf_bars"), width = 5),
        box(htmlOutput("tfidfExplanation"), background = 'red', width = 5)
      ),
    )
  )
)