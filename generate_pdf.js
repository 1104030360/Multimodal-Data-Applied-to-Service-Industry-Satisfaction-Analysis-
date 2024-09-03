const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');

(async () => {
    try {
        const browser = await puppeteer.launch();
        const page = await browser.newPage();
        const pdf_folder = 'static/pdf';
        const pdf_filename = 'report1.pdf';

        // 确保目录存在
        const pdfDirectory = path.resolve(__dirname, pdf_folder);
        console.log('PDF Directory:', pdfDirectory);  // 打印目录路径

        if (!fs.existsSync(pdfDirectory)) {
            fs.mkdirSync(pdfDirectory, { recursive: true });
            console.log('Directory created:', pdfDirectory);  // 打印创建的目录路径
        }

        // 确保保存路径正确
        const pdfPath = path.join(pdfDirectory, pdf_filename);
        console.log('Saving PDF to:', pdfPath);  // 打印完整的保存路径

        // 加载你的 Flask 应用页面
        await page.goto('http://127.0.0.1:5000/', { waitUntil: 'networkidle2', timeout: 0 });

        // 如果页面内容很多，可以设置页面高度来生成多页PDF
        const desiredHeight = '297mm'; // 这是A4纸的标准高度，适用于单页
        // 如果你的内容超过一页，可以尝试设置更大的值，比如 '1000mm'
        await page.pdf({
            path: pdfPath,
            format: 'A4',  // 使用 A4 纸格式
            printBackground: true,
            height: desiredHeight,  // 设置页面的高度
        });

        console.log('PDF generated successfully at', pdfPath);  // 确认PDF生成成功并显示路径
        await browser.close();
    } catch (error) {
        console.error('Error generating PDF:', error);
    }
})();
